# -*- coding: utf-8 -*-
# NormalMapToMesh 操作层 —— bpy 侧: 直算前端(免烘焙)、Multires 管理、
# SIMPLE 细分求值、numpy 位移、multires_reshape 写回。
#
# v6 不变式: 一切场量只由低模 + 法线贴图决定, 细分只是采样密度——任何级别
# 都输出同一张曲面(基面 = 低模的确定函数), 位移管线不读细分网格自身属性。
#   1. 直算前端: mikktspace 切线帧(calc_tangents, 与渲染器同源)+逐三角形解析
#      ∂P/∂u,∂P/∂v 光栅化到 UV 网格; 材质法线链用 numpy 节点求值器直接算,
#      切线空间法线图 + 切线帧 → 高度梯度。不支持的节点/网格回退 EMIT 三图烘焙。
#   2. 镜像 Neumann 泊松积分 → 物理高度场(平贴严格 0) → 开放边界距离衰减。
#   3. Multires 建层只为数据结构(隐藏态跑 subdivide; reshape 完整覆写 MDISPS);
#      重建时层数匹配则整段跳过。
#   4. 细分基面 = 岛界折痕锁定的 Catmull-Clark 极限曲面(Subsurf 求值副本,
#      use_limit_surface: 任何级别都是同一极限曲面的嵌套采样): 低模粗曲率由
#      细分平滑(法线图只携带高频细节), 而所有 UV 岛边界边 crease=1 + 岛界顶点
#      vertex crease=1——折痕链逐级取中点且端点钉死, 边界折线被精确锁在原位
#      (CC 默认把边界链平滑成 B 样条曲线, 即"边缘软化"/卡片缝隙的根源)。
#      UV 与低模角法线(NMTM_N0 corner 属性)保持线性插值(uv_smooth=NONE),
#      位移方向即 shader 的逐像素插值法线场, 采样对位不随细分漂移。
#   5. 采样高度(B 样条 C2) → 逐岛去趋势/缝合 → 边界硬锁 → 位移 → reshape 写回。
#
# 重复点"应用"= 从基面重建(幂等), 改倍数即所见即所得(全缓存命中, 只剩写回)。

import time

import bpy
import numpy as np
from bpy_extras.io_utils import ImportHelper

from . import core

MAX_LEVELS = 9
BAKE_MARGIN = 16
BAKE_SAMPLES = 1   # EMIT 烘焙无噪声, 1 采样足够(回退路径)

# 运行期缓存(只认网格/材质"身份", 不追踪材质节点内容变化)
_grad_cache = {}     # 前端结果: (gx, gy, wmap)
_island_cache = {}


def _cache_put(cache, key, val, cap=2):
    cache.pop(key, None)
    while len(cache) >= cap:
        cache.pop(next(iter(cache)))
    cache[key] = val


def clear_caches():
    _grad_cache.clear()
    _island_cache.clear()


# ---------------------------------------------------------------------------
# 数据读取(全 foreach_get, 零 Python 逐元素循环)
# ---------------------------------------------------------------------------

def _read_loop_uvs(me):
    """兼容 3.5+ 的 layer.uv 与旧 layer.data 两种访问路径。"""
    layer = me.uv_layers.active
    if layer is None:
        raise RuntimeError("网格没有活动 UV 层")
    n = len(me.loops)
    buf = np.empty(n * 2, np.float32)
    try:
        layer.uv.foreach_get("vector", buf)
    except (AttributeError, TypeError):
        layer.data.foreach_get("uv", buf)
    return buf.reshape(-1, 2)


def _read_loop_verts(me):
    n = len(me.loops)
    buf = np.empty(n, np.int32)
    me.loops.foreach_get("vertex_index", buf)
    return buf


def _read_vert_cos(me):
    n = len(me.vertices)
    buf = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", buf)
    return buf.reshape(-1, 3)


def _uv_fill(me, loop_uv):
    """UV 占用率(自动级别用): 三角化 UV 面积之和, 裁到 [0.02, 1]。"""
    me.calc_loop_triangles()
    t = len(me.loop_triangles)
    tl = np.empty(t * 3, np.int32)
    me.loop_triangles.foreach_get("loops", tl)
    uvt = loop_uv[tl].reshape(-1, 3, 2)
    e1 = uvt[:, 1] - uvt[:, 0]
    e2 = uvt[:, 2] - uvt[:, 0]
    uv_area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
    return float(min(max(uv_area, 0.02), 1.0))


# ---------------------------------------------------------------------------
# 法线来源判定
# ---------------------------------------------------------------------------

def _tree_has_normal_map(nt, seen=None):
    if nt is None:
        return False
    if seen is None:
        seen = set()
    if nt.name_full in seen:
        return False
    seen.add(nt.name_full)
    for n in nt.nodes:
        if n.type == 'NORMAL_MAP':
            return True
        if n.type == 'GROUP' and _tree_has_normal_map(n.node_tree, seen):
            return True
    return False


def _has_material_normal_chain(obj):
    return any(s.material is not None and _tree_has_normal_map(s.material.node_tree)
               for s in obj.material_slots)


def _resolve_source(obj, s):
    """返回 'MATERIAL' 或 'IMAGE'。AUTO 优先材质自带法线链(最忠实, 含通道重建网络)。"""
    if s.source == 'MATERIAL':
        if not _has_material_normal_chain(obj):
            raise RuntimeError("物体材质里没有 Normal Map 节点, 无法按材质求值; 请改用贴图模式")
        return 'MATERIAL'
    if s.source == 'IMAGE':
        if s.image is None:
            raise RuntimeError("贴图模式需要先选择法线贴图")
        return 'IMAGE'
    if _has_material_normal_chain(obj):
        return 'MATERIAL'
    if s.image is not None:
        return 'IMAGE'
    raise RuntimeError("物体材质没有 Normal Map 节点, 也没有选择贴图——两者需有其一")


def _upstream_image_sizes(start_socket, seen=None):
    """从某 socket 沿输入连线向上游 BFS, 收集经过的 Image Texture 节点分辨率。"""
    if seen is None:
        seen = set()
    sizes = []
    stack = [start_socket]
    while stack:
        sock = stack.pop()
        if not sock.is_linked:
            continue
        node = sock.links[0].from_node
        key = (node.id_data.name_full, node.name)
        if key in seen:
            continue
        seen.add(key)
        if node.type == 'TEX_IMAGE' and node.image is not None:
            w, h = node.image.size
            if w > 0 and h > 0:
                sizes.append(max(w, h))
        # 节点组内部不展开(求值器本身也不支持 GROUP, 命中即整体回退烘焙路径;
        # 分辨率探测保守跳过, 不影响正确性, 只影响自动选到的工作分辨率)
        for inp in node.inputs:
            stack.append(inp)
    return sizes


def _native_resolution(obj, source, image):
    """工作分辨率 = 实际接入的法线贴图原生分辨率, 不再由用户猜数字。

    高于源贴图分辨率的网格只是把已有像素插值放大, 不产生任何新细节还多耗算力;
    低于源分辨率则白白丢弃作者烘焙进贴图的信息。两者都没有意义, 直接对齐现实。
    MATERIAL 来源沿每个材质 Normal Map 节点的 Color 输入网络回溯, 取所有材质槽
    命中的最大分辨率(多材质共享同一张工作网格); 找不到则回退 2048。
    """
    if source == 'IMAGE':
        w, h = image.size
        return max(w, h, 64)
    sizes = []
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or mat.node_tree is None:
            continue
        for n in mat.node_tree.nodes:
            if n.type == 'NORMAL_MAP':
                sizes.extend(_upstream_image_sizes(n.inputs['Color']))
    if not sizes:
        print("[NormalMapToMesh] 警告: 未在材质法线链中找到贴图, 工作分辨率回退 2048")
        return 2048
    return max(sizes)


# ---------------------------------------------------------------------------
# 直算前端: numpy 材质法线链求值器
# ---------------------------------------------------------------------------

class _NodeEvalUnsupported(Exception):
    """材质网络含求值器不支持的节点/接法 → 整体回退 Cycles 烘焙路径。"""


def _read_image_grid(img, size):
    """图像重采样到 (size, size, 4): 与烘焙语义一致——网格 texel 中心做双线性。
    分辨率恰好相同时为逐位直读。"""
    if img is None:
        raise _NodeEvalUnsupported("图像节点没有图像")
    w, h = img.size
    if w == 0 or h == 0:
        raise RuntimeError(f"贴图 '{img.name}' 没有像素数据(文件缺失?)")
    ch = img.channels
    buf = np.empty(w * h * ch, np.float32)
    img.pixels.foreach_get(buf)
    px = buf.reshape(h, w, ch)
    if ch < 4:
        rgba = np.ones((h, w, 4), np.float32)
        rgba[..., :ch] = px
        px = rgba
    if (w, h) == (size, size):
        return px[..., :4]
    uu = ((np.arange(size) + 0.5) / size).astype(np.float32)
    grid_u = np.broadcast_to(uu[None, :], (size, size)).ravel()
    grid_v = np.broadcast_to(uu[:, None], (size, size)).ravel()
    out = core.sample_bilinear_wrap(px[..., :4], grid_u, grid_v)
    return out.reshape(size, size, 4).astype(np.float32)


def _as_scalar(x, size):
    if isinstance(x, np.ndarray):
        if x.ndim == 3:
            # 颜色隐转标量: 取平均(Blender 隐转是亮度, 法线链里几乎不出现——保守拒绝)
            raise _NodeEvalUnsupported("颜色→标量隐式转换")
        return x
    return float(x)


def _as_vec3(x, size):
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            return np.repeat(x[..., None], 3, axis=2)
        return x[..., :3]
    if isinstance(x, (int, float)):
        return np.full((size, size, 3), float(x), np.float32)
    v = np.asarray(x, np.float32)[:3]
    return np.broadcast_to(v, (size, size, 3)).copy()


def _eval_socket(socket, size, memo):
    """递归求值输出 socket → float / (S,S) / (S,S,3)。不支持 → _NodeEvalUnsupported。"""
    key = (socket.node.name, socket.identifier)
    if key in memo:
        return memo[key]
    node = socket.node
    nt = node.type

    def inp(i):
        sk = node.inputs[i]
        if sk.is_linked:
            return _eval_socket(sk.links[0].from_socket, size, memo)
        dv = sk.default_value
        try:
            return float(dv)
        except TypeError:
            return tuple(dv)[:3]

    if nt == 'REROUTE':
        val = _eval_socket(node.inputs[0].links[0].from_socket, size, memo) \
            if node.inputs[0].is_linked else 0.0
    elif nt == 'TEX_IMAGE':
        if node.inputs['Vector'].is_linked:
            raise _NodeEvalUnsupported("图像节点带自定义 Vector 输入")
        rgba = _read_image_grid(node.image, size)
        if socket.name == 'Alpha':
            val = rgba[..., 3].copy()
        else:
            val = rgba[..., :3].copy()
    elif nt in ('SEPARATE_COLOR', 'SEPRGB', 'SEPARATE_XYZ', 'SEPXYZ'):
        if nt == 'SEPARATE_COLOR' and getattr(node, 'mode', 'RGB') != 'RGB':
            raise _NodeEvalUnsupported(f"Separate Color 模式 {node.mode}")
        vec = _as_vec3(inp(0), size)
        idx = {'Red': 0, 'Green': 1, 'Blue': 2, 'X': 0, 'Y': 1, 'Z': 2}[socket.name]
        val = vec[..., idx].copy()
    elif nt in ('COMBINE_COLOR', 'COMBRGB', 'COMBINE_XYZ', 'COMBXYZ'):
        if nt == 'COMBINE_COLOR' and getattr(node, 'mode', 'RGB') != 'RGB':
            raise _NodeEvalUnsupported(f"Combine Color 模式 {node.mode}")
        parts = [_as_scalar(inp(i), size) for i in range(3)]
        if all(isinstance(p, float) for p in parts):
            val = tuple(parts)
        else:
            parts = [p if isinstance(p, np.ndarray)
                     else np.full((size, size), p, np.float32) for p in parts]
            val = np.stack(parts, axis=-1).astype(np.float32)
    elif nt == 'MATH':
        op = node.operation
        a = _as_scalar(inp(0), size)
        b = _as_scalar(inp(1), size) if len(node.inputs) > 1 else 0.0
        if op == 'ADD':
            val = a + b
        elif op == 'SUBTRACT':
            val = a - b
        elif op == 'MULTIPLY':
            val = a * b
        elif op == 'DIVIDE':
            val = a / np.maximum(np.abs(b), 1e-20) * np.sign(b) if isinstance(b, np.ndarray) \
                else (a / b if b != 0.0 else a * 0.0)
        elif op == 'MULTIPLY_ADD':
            val = a * b + _as_scalar(inp(2), size)
        elif op == 'POWER':
            val = np.power(np.maximum(a, 0.0), b) if isinstance(a, np.ndarray) else a ** b
        elif op == 'SQRT':
            val = np.sqrt(np.maximum(a, 0.0))
        elif op == 'ABSOLUTE':
            val = np.abs(a)
        elif op == 'MINIMUM':
            val = np.minimum(a, b)
        elif op == 'MAXIMUM':
            val = np.maximum(a, b)
        elif op == 'FLOOR':
            val = np.floor(a)
        elif op == 'ROUND':
            val = np.round(a)
        elif op == 'FRACT':
            val = a - np.floor(a)
        else:
            raise _NodeEvalUnsupported(f"Math 运算 {op}")
        if node.use_clamp:
            val = np.clip(val, 0.0, 1.0)
    elif nt == 'VECT_MATH':
        op = node.operation
        a = _as_vec3(inp(0), size)
        b = _as_vec3(inp(1), size) if len(node.inputs) > 1 else None
        if op == 'ADD':
            val = a + b
        elif op == 'SUBTRACT':
            val = a - b
        elif op == 'MULTIPLY':
            val = a * b
        elif op == 'DIVIDE':
            val = a / np.where(np.abs(b) < 1e-20, 1.0, b)
        elif op == 'MULTIPLY_ADD':
            val = a * b + _as_vec3(inp(2), size)
        elif op == 'SCALE':
            sc = node.inputs['Scale']
            scv = _eval_socket(sc.links[0].from_socket, size, memo) if sc.is_linked \
                else float(sc.default_value)
            val = a * (scv[..., None] if isinstance(scv, np.ndarray) else scv)
        elif op == 'NORMALIZE':
            ln = np.linalg.norm(a, axis=-1, keepdims=True)
            val = a / np.maximum(ln, 1e-20)
        elif op == 'DOT_PRODUCT':
            val = np.einsum('...i,...i->...', a, b).astype(np.float32)
        elif op == 'CROSS_PRODUCT':
            val = np.cross(a, b).astype(np.float32)
        elif op == 'LENGTH':
            val = np.linalg.norm(a, axis=-1).astype(np.float32)
        else:
            raise _NodeEvalUnsupported(f"Vector Math 运算 {op}")
        if isinstance(val, np.ndarray) and socket.name == 'Value' and val.ndim == 3:
            raise _NodeEvalUnsupported(f"Vector Math {op} 的 Value 输出")
    elif nt == 'VALUE':
        val = float(node.outputs[0].default_value)
    elif nt == 'RGB':
        val = tuple(node.outputs[0].default_value)[:3]
    elif nt == 'GAMMA':
        a = _as_vec3(inp(0), size)
        g = _as_scalar(inp(1), size)
        val = np.power(np.maximum(a, 0.0), g)
    elif nt == 'INVERT':
        fac = _as_scalar(inp(0), size)
        col = _as_vec3(inp(1), size)
        val = col + (1.0 - 2.0 * col) * (fac[..., None] if isinstance(fac, np.ndarray) else fac)
    else:
        raise _NodeEvalUnsupported(f"节点类型 {nt}")
    memo[key] = val
    return val


def _eval_material_tangent_map(mat, size):
    """numpy 求值材质法线链 → (S,S,3) 切线空间法线(已解码, 含 Strength)。

    取第一个 Normal Map 节点的 Color 输入上游网络求值, t = 2c−1;
    Strength ≠ 1 时 t' = (0,0,1)(1−s) + t·s (Normal Map 节点的线性混合语义)。
    无 Normal Map 节点 → None(平坦)。
    """
    nt_tree = mat.node_tree
    nmaps = [n for n in nt_tree.nodes if n.type == 'NORMAL_MAP']
    if not nmaps:
        return None
    if len(nmaps) > 1:
        print(f"[NormalMapToMesh] 警告: 材质 '{mat.name}' 有 {len(nmaps)} 个 Normal Map, 取第一个")
    nmap = nmaps[0]
    if nmap.space != 'TANGENT':
        raise _NodeEvalUnsupported(f"Normal Map 空间 {nmap.space}")
    if nmap.inputs['Strength'].is_linked:
        raise _NodeEvalUnsupported("Normal Map Strength 被连线")
    strength = float(nmap.inputs['Strength'].default_value)
    csock = nmap.inputs['Color']
    if not csock.is_linked:
        col = np.broadcast_to(np.array([0.5, 0.5, 1.0], np.float32), (size, size, 3)).copy()
    else:
        col = _as_vec3(_eval_socket(csock.links[0].from_socket, size, {}), size)
    t = col.astype(np.float32) * 2.0 - 1.0
    if strength != 1.0:
        flat = np.array([0.0, 0.0, 1.0], np.float32)
        t = flat * (1.0 - strength) + t * strength
    return t


# ---------------------------------------------------------------------------
# 直算前端: 切线帧光栅化 + 梯度
# ---------------------------------------------------------------------------

def _mesh_fingerprint(me, loop_uv):
    fp = hash(loop_uv[:: max(1, loop_uv.shape[0] // 4096)].tobytes())
    return (me.name_full, len(me.vertices), len(me.polygons), len(me.loops), fp)


def _gradients_direct(obj, me, source, image, size, loop_uv, loop_vert,
                      deadzone=0.0, slope_limit=0.0):
    """免烘焙直算: mikktspace 切线帧标量 + 逐三角形解析 ∂P → UV 高度梯度。

    shader 等价求值纪律——几何属性不进"赢家覆盖"的共享网格:
    重叠 UV 卡片(正/背面、镜像复用、图集多层)会让逐 texel 覆盖形成
    逐三角形补丁的属性马赛克, 正/背面梯度互为相反数, 积分后成严重锯齿。
    因此: ①梯度帧标量只由**正 UV 绕向**三角形贡献且逐 texel **平均**
    (孤儿镜像岛用负绕向做二次补洞); ②位移方向不进网格, 由消费端取
    各 loop 自己网格的平滑角法线。返回 (gx, gy, wmap)。
    """
    uv_name = me.uv_layers.active.name
    try:
        me.calc_tangents(uvmap=uv_name)
    except RuntimeError as e:
        raise _NodeEvalUnsupported(f"calc_tangents 失败(网格含五边以上面?): {e}")
    n_l = len(me.loops)
    tan = np.empty(n_l * 3, np.float32)
    me.loops.foreach_get("tangent", tan)
    tan = tan.reshape(-1, 3)
    sign = np.empty(n_l, np.float32)
    me.loops.foreach_get("bitangent_sign", sign)
    nrm = np.empty(n_l * 3, np.float32)
    me.corner_normals.foreach_get("vector", nrm)
    nrm = nrm.reshape(-1, 3)
    me.free_tangents()

    pos = _read_vert_cos(me)
    me.calc_loop_triangles()
    t_count = len(me.loop_triangles)
    tl = np.empty(t_count * 3, np.int32)
    me.loop_triangles.foreach_get("loops", tl)
    tl = tl.reshape(-1, 3)
    tp = np.empty(t_count, np.int32)
    me.loop_triangles.foreach_get("polygon_index", tp)
    pmat = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("material_index", pmat)

    tri_uv = loop_uv[tl.ravel()].reshape(-1, 3, 2)
    tri_pos = pos[loop_vert[tl.ravel()]].reshape(-1, 3, 3)
    d1 = (tri_uv[:, 1] - tri_uv[:, 0]).astype(np.float64)
    d2 = (tri_uv[:, 2] - tri_uv[:, 0]).astype(np.float64)
    e1 = (tri_pos[:, 1] - tri_pos[:, 0]).astype(np.float64)
    e2 = (tri_pos[:, 2] - tri_pos[:, 0]).astype(np.float64)
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    valid = np.abs(det) > 1e-16
    det_safe = np.where(valid, det, 1.0)
    pu = (e1 * d2[:, 1, None] - e2 * d1[:, 1, None]) / det_safe[:, None]
    pv = (e2 * d1[:, 0, None] - e1 * d2[:, 0, None]) / det_safe[:, None]

    # 逐角切线帧标量: au=T·Pu, bu=B·Pu, av=T·Pv, bv=B·Pv (B = sign·N×T)
    tc = tan[tl.ravel()].reshape(-1, 3, 3).astype(np.float64)
    nc = nrm[tl.ravel()].reshape(-1, 3, 3).astype(np.float64)
    sc = sign[tl.ravel()].reshape(-1, 3).astype(np.float64)
    bc = np.cross(nc, tc) * sc[:, :, None]
    attrs = np.empty((t_count, 3, 4), np.float32)
    attrs[:, :, 0] = np.einsum('tcj,tj->tc', tc, pu)
    attrs[:, :, 1] = np.einsum('tcj,tj->tc', bc, pu)
    attrs[:, :, 2] = np.einsum('tcj,tj->tc', tc, pv)
    attrs[:, :, 3] = np.einsum('tcj,tj->tc', bc, pv)

    # 正绕向为主贡献(逐 texel 平均), 负绕向只补正绕向没覆盖的洞
    pos_sel = valid & (det > 0)
    neg_sel = valid & (det < 0)
    sum_p, cnt_p = core.rasterize_tris(tri_uv[pos_sel], attrs[pos_sel], size,
                                       accumulate=True)
    frame = np.zeros((size, size, 4), np.float32)
    covered_p = cnt_p > 0
    frame[covered_p] = sum_p[covered_p] / cnt_p[covered_p][:, None]
    mask0 = covered_p
    if neg_sel.any():
        sum_n, cnt_n = core.rasterize_tris(tri_uv[neg_sel], attrs[neg_sel], size,
                                           accumulate=True)
        fill = (~covered_p) & (cnt_n > 0)
        if fill.any():
            frame[fill] = sum_n[fill] / cnt_n[fill][:, None]
            mask0 = covered_p | fill

    # 切线空间法线图
    if source == 'IMAGE':
        t_map = _read_image_grid(image, size)[..., :3] * 2.0 - 1.0
    else:
        flat = np.broadcast_to(np.array([0.0, 0.0, 1.0], np.float32), (size, size, 3))
        mat_maps = []
        for slot in obj.material_slots:
            m = slot.material
            t = _eval_material_tangent_map(m, size) if m is not None else None
            mat_maps.append(flat if t is None else t)
        if not mat_maps:
            mat_maps = [flat]
        if len(mat_maps) == 1:
            t_map = np.ascontiguousarray(mat_maps[0])
        else:
            # 多材质槽: 逐 texel 材质号(覆盖式光栅化)选择对应贴图链结果
            mat_attr = np.broadcast_to(
                pmat[tp].astype(np.float32)[:, None, None], (t_count, 3, 1)).copy()
            mgrid, _ = core.rasterize_tris(tri_uv[valid], mat_attr[valid], size)
            mi = np.clip(np.round(mgrid[..., 0]).astype(np.int64), 0, len(mat_maps) - 1)
            t_map = np.empty((size, size, 3), np.float32)
            for i, m in enumerate(mat_maps):
                sel = mi == i
                t_map[sel] = m[sel]

    # 梯度只取真实 UV 覆盖区: 外扩 margin 的复制内容会虚增积分能量
    gx, gy, _ = core.gradients_from_frame_scalars(
        t_map, frame[..., 0], frame[..., 1], frame[..., 2], frame[..., 3], mask0,
        deadzone=deadzone, slope_limit=slope_limit)

    # 采样有效域: 掩码外扩(岛边界 B 样条采样不吃到无效 texel)
    wmap = core.dilate_mask(mask0, BAKE_MARGIN).astype(np.float32)
    return gx, gy, wmap


def _gradients_cached(context, obj, me, source, image, size, loop_uv, loop_vert,
                      force_bake, deadzone, slope_limit):
    mats = tuple(s.material.name_full if s.material else '' for s in obj.material_slots)
    key = (_mesh_fingerprint(me, loop_uv), mats, source,
           image.name_full if image is not None else '', int(size), bool(force_bake),
           round(float(deadzone), 6), round(float(slope_limit), 6))
    got = _grad_cache.get(key)
    if got is not None:
        return got

    result = None
    if not force_bake:
        try:
            result = _gradients_direct(obj, me, source, image, size, loop_uv, loop_vert,
                                       deadzone=deadzone, slope_limit=slope_limit)
            print("[NormalMapToMesh] 前端: 直算(免烘焙)")
        except _NodeEvalUnsupported as e:
            print(f"[NormalMapToMesh] 直算不支持({e}), 回退 Cycles 烘焙")
    if result is None:
        rgb_detail, rgb_base, pos_map = _bake_triple(context, obj, source, image, size, loop_uv)
        gx, gy, wmap = core.height_gradients(rgb_detail, rgb_base, pos_map,
                                             deadzone=deadzone, slope_limit=slope_limit)
        result = (gx, gy, wmap)
    _cache_put(_grad_cache, key, result)
    return result


# ---------------------------------------------------------------------------
# 回退路径: Cycles EMIT 三图烘焙 (v3 原样保留)
# ---------------------------------------------------------------------------

def _build_encoder(nt, src_socket, vector_type='NORMAL', encode=True):
    """向量(世界空间) → Vector Transform 转物体空间 → (可选 ×0.5+0.5 编码) → Emission。"""
    vt = nt.nodes.new('ShaderNodeVectorTransform')
    vt.vector_type = vector_type
    vt.convert_from = 'WORLD'
    vt.convert_to = 'OBJECT'
    nt.links.new(vt.inputs['Vector'], src_socket)
    nodes = [vt]
    out_socket = vt.outputs['Vector']
    if encode:
        vm = nt.nodes.new('ShaderNodeVectorMath')
        vm.operation = 'MULTIPLY_ADD'
        vm.inputs[1].default_value = (0.5, 0.5, 0.5)
        vm.inputs[2].default_value = (0.5, 0.5, 0.5)
        nt.links.new(vm.inputs[0], out_socket)
        out_socket = vm.outputs['Vector']
        nodes.append(vm)
    em = nt.nodes.new('ShaderNodeEmission')
    nt.links.new(em.inputs['Color'], out_socket)
    nodes.append(em)
    return nodes, em


def _bake_once(context, obj, kind, source, image, bake_size):
    """单次物体空间 EMIT 烘焙 → (H, W, 3) float32。(回退路径)"""
    me = obj.data
    bake_img = None
    tmp_mat = None
    saved_slots = None
    slot_appended = False
    inserted = []
    grafts = []
    try:
        bake_img = bpy.data.images.new(f"NMTM_bake_{kind}", width=bake_size,
                                       height=bake_size, float_buffer=True)
        bake_img.colorspace_settings.name = 'Non-Color'

        if kind != 'DETAIL' or source == 'IMAGE':
            tmp_mat = bpy.data.materials.new("NMTM_bake_mat")
            nt = tmp_mat.node_tree
            nt.nodes.clear()
            out = nt.nodes.new('ShaderNodeOutputMaterial')
            if kind == 'DETAIL':
                timg = nt.nodes.new('ShaderNodeTexImage')
                timg.image = image
                nmap = nt.nodes.new('ShaderNodeNormalMap')
                nt.links.new(nmap.inputs['Color'], timg.outputs['Color'])
                src_socket, vtype, enc = nmap.outputs['Normal'], 'NORMAL', True
            elif kind == 'BASELINE':
                geo = nt.nodes.new('ShaderNodeNewGeometry')
                src_socket, vtype, enc = geo.outputs['Normal'], 'NORMAL', True
            else:   # POSITION
                geo = nt.nodes.new('ShaderNodeNewGeometry')
                src_socket, vtype, enc = geo.outputs['Position'], 'POINT', False
            _, em = _build_encoder(nt, src_socket, vtype, enc)
            nt.links.new(out.inputs['Surface'], em.outputs['Emission'])
            for n in nt.nodes:
                n.select = False
            target = nt.nodes.new('ShaderNodeTexImage')
            target.image = bake_img
            target.select = True
            nt.nodes.active = target
            if obj.material_slots:
                saved_slots = [s.material for s in obj.material_slots]
                for i in range(len(obj.material_slots)):
                    obj.material_slots[i].material = tmp_mat
            else:
                me.materials.append(tmp_mat)
                slot_appended = True
        else:
            done = set()
            for slot in obj.material_slots:
                mat = slot.material
                if mat is None or mat.name in done:
                    continue
                done.add(mat.name)
                nt = mat.node_tree
                out_node = nt.get_output_node('CYCLES')
                if out_node is None:
                    continue
                surf = out_node.inputs['Surface']
                orig_from = None
                if surf.is_linked:
                    lk = surf.links[0]
                    orig_from = (lk.from_node.name, lk.from_socket.name)
                nmap_names = [n.name for n in nt.nodes if n.type == 'NORMAL_MAP']
                if len(nmap_names) > 1:
                    print(f"[NormalMapToMesh] 警告: 材质 '{mat.name}' 有 "
                          f"{len(nmap_names)} 个 Normal Map 节点, 取第一个")
                new_nodes = []
                if nmap_names:
                    src_socket = nt.nodes[nmap_names[0]].outputs['Normal']
                else:
                    geo = nt.nodes.new('ShaderNodeNewGeometry')
                    new_nodes.append(geo)
                    src_socket = geo.outputs['Normal']
                enc_nodes, em = _build_encoder(nt, src_socket)
                new_nodes.extend(enc_nodes)
                nt.links.new(out_node.inputs['Surface'], em.outputs['Emission'])
                grafts.append((mat.name, [n.name for n in new_nodes],
                               out_node.name, orig_from))

                prev_active = nt.nodes.active.name if nt.nodes.active else ''
                # bpy 集合迭代出的是新包装对象, `is` 比较恒假——先全清再对持有的原始引用赋值
                for n in nt.nodes:
                    n.select = False
                target = nt.nodes.new('ShaderNodeTexImage')
                target.image = bake_img
                target.location = (0, 600)
                target.select = True
                nt.nodes.active = target
                inserted.append((mat.name, target.name, prev_active))

        bpy.ops.object.bake(type='EMIT')

        buf = np.empty(bake_size * bake_size * 4, np.float32)
        bake_img.pixels.foreach_get(buf)
        rgb = buf.reshape(bake_size, bake_size, 4)[..., :3].astype(np.float32, copy=True)
        if float(rgb.std()) < 1e-5:
            raise RuntimeError(f"{kind} 烘焙结果是纯色, 物体空间法线烘焙未生效(检查材质与 UV)")
        return rgb
    finally:
        for mat_name, node_names, out_name, orig_from in grafts:
            mat = bpy.data.materials.get(mat_name)
            if mat is None:
                continue
            nt = mat.node_tree
            for nn in node_names:
                n = nt.nodes.get(nn)
                if n is not None:
                    nt.nodes.remove(n)
            if orig_from is not None:
                out_node = nt.nodes.get(out_name)
                from_node = nt.nodes.get(orig_from[0])
                if out_node is not None and from_node is not None:
                    try:
                        nt.links.new(out_node.inputs['Surface'],
                                     from_node.outputs[orig_from[1]])
                    except Exception:
                        pass
        for mat_name, node_name, prev_active in inserted:
            mat = bpy.data.materials.get(mat_name)
            if mat is None:
                continue
            nt = mat.node_tree
            n = nt.nodes.get(node_name)
            if n is not None:
                nt.nodes.remove(n)
            if prev_active:
                pa = nt.nodes.get(prev_active)
                if pa is not None:
                    nt.nodes.active = pa
        if saved_slots is not None:
            for i, m in enumerate(saved_slots):
                obj.material_slots[i].material = m
        if slot_appended:
            me.materials.pop(index=len(me.materials) - 1)
        if tmp_mat is not None:
            bpy.data.materials.remove(tmp_mat)
        if bake_img is not None:
            bpy.data.images.remove(bake_img)


def _bake_triple(context, obj, source, image, bake_size, loop_uv):
    """三次同参数物体空间 EMIT 烘焙(n1, n0, P)。(回退路径, 无缓存——由上层缓存)"""
    if source == 'MATERIAL':
        for slot in obj.material_slots:
            if slot.material is not None and slot.material.library is not None:
                raise RuntimeError(
                    f"材质 '{slot.material.name}' 来自链接库, 无法插入烘焙节点; 请先 Make Local")
    if source == 'IMAGE':
        try:
            if image.source == 'FILE' and image.colorspace_settings.name != 'Non-Color':
                image.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass

    scene = context.scene
    saved_scene = (scene.render.engine, scene.cycles.device, scene.cycles.samples,
                   scene.cycles.bake_type, scene.render.bake.use_selected_to_active,
                   scene.render.bake.margin)
    saved_hide = [(o.name, o.hide_render) for o in bpy.data.objects]
    saved_show_render = [(m.name, m.show_render) for m in obj.modifiers]
    try:
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = BAKE_SAMPLES
        scene.cycles.bake_type = 'EMIT'
        scene.render.bake.use_selected_to_active = False
        scene.render.bake.margin = BAKE_MARGIN

        for o in bpy.data.objects:
            o.hide_render = True
        obj.hide_render = False
        for o in list(context.selected_objects):
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        for m in obj.modifiers:
            m.show_render = False

        rgb_detail = _bake_once(context, obj, 'DETAIL', source, image, bake_size)
        rgb_base = _bake_once(context, obj, 'BASELINE', source, None, bake_size)
        pos_map = _bake_once(context, obj, 'POSITION', source, None, bake_size)
    finally:
        for name, vis in saved_hide:
            o = bpy.data.objects.get(name)
            if o is not None:
                o.hide_render = vis
        for name, vis in saved_show_render:
            m = obj.modifiers.get(name)
            if m is not None:
                m.show_render = vis
        (scene.render.engine, scene.cycles.device, scene.cycles.samples,
         scene.cycles.bake_type, scene.render.bake.use_selected_to_active,
         scene.render.bake.margin) = saved_scene

    return rgb_detail, rgb_base, pos_map


# ---------------------------------------------------------------------------
# 岛界折痕锁定的 Catmull-Clark 极限曲面求值(拓扑与 multires 逐位一致, 实测)
# ---------------------------------------------------------------------------

def _island_border_edges(me, loop_uv, loop_vert, labels, loop_total):
    """UV 岛边界边集合 + 其端点集合(折痕锁定目标)。

    边界 = 开放边/非流形边、两侧面属不同岛的边、两侧 UV 不连续的接缝边
    (量化口径与 face_islands 一致)。返回 (edge_bool[E], vert_bool[V])。
    """
    e_count = len(me.edges)
    l_count = len(me.loops)
    le = np.empty(l_count, np.int32)
    me.loops.foreach_get("edge_index", le)
    ls = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_start", ls)
    lt = loop_total.astype(np.int64)
    nxt = np.arange(l_count, dtype=np.int64) + 1
    nxt[ls + lt - 1] = ls
    poly_of_loop = np.repeat(np.arange(len(me.polygons), dtype=np.int64), lt)

    counts = np.bincount(le, minlength=e_count)
    border = counts != 2                       # 开放边/非流形边一律锁
    two_edges = np.flatnonzero(counts == 2)
    if two_edges.size:
        order = np.argsort(le, kind='stable')
        first = np.searchsorted(le[order], two_edges, side='left')
        l1 = order[first].astype(np.int64)
        l2 = order[first + 1].astype(np.int64)
        diff_isl = labels[poly_of_loop[l1]] != labels[poly_of_loop[l2]]
        quv = np.round(loop_uv.astype(np.float64) * 65536.0).astype(np.int64)
        lv = loop_vert.astype(np.int64)
        opp = lv[l2] != lv[l1]                 # 对向绕行(流形正常态)
        c2a = np.where(opp, nxt[l2], l2)       # 对侧面上与 l1 同顶点的角
        c2b = np.where(opp, l2, nxt[l2])
        seam = ((quv[l1] != quv[c2a]).any(axis=1)
                | (quv[nxt[l1]] != quv[c2b]).any(axis=1))
        border[two_edges[diff_isl | seam]] = True

    ev = np.empty(e_count * 2, np.int32)
    me.edges.foreach_get("vertices", ev)
    vert_pin = np.zeros(len(me.vertices), bool)
    vert_pin[ev.reshape(-1, 2)[border].ravel()] = True
    return border, vert_pin


def _subsurf_eval_mesh(context, obj, level, border_edges, border_verts):
    """网格副本 + 折痕锁定 CC Subsurf 极限求值 → 新 Mesh(调用方负责删除)。

    无算子、无选择/撤销依赖, 也天然不受 Mesh 里已有 MDISPS 影响(Subsurf 忽略之)。
    岛界边 crease=1 + 岛界顶点 vertex crease=1: 折痕链逐级取线性中点、原顶点
    钉死——边界折线精确保持原位(CC 默认把边界链平滑成 B 样条曲线 = 边缘软化);
    内部收敛到 C2 极限曲面, use_limit_surface 使任何级别都是同一曲面的嵌套采样。
    副本上把低模平滑角法线写成 corner 属性 NMTM_N0, uv_smooth=NONE 保证 UV 与
    该属性均为纯线性插值(shader 重心插值同构), 全程零高模数据。与用户已有
    折痕取 max 合并, 原网格不动。
    """
    me2 = obj.data.copy()
    nrm = np.empty(len(me2.loops) * 3, np.float32)
    me2.corner_normals.foreach_get("vector", nrm)
    attr = me2.attributes.new("NMTM_N0", 'FLOAT_VECTOR', 'CORNER')
    attr.data.foreach_set("vector", nrm)

    ec = me2.attributes.get("crease_edge")
    if ec is None or ec.domain != 'EDGE':
        ec = me2.attributes.new("crease_edge", 'FLOAT', 'EDGE')
        cur_e = np.zeros(len(me2.edges), np.float32)
    else:
        cur_e = np.empty(len(me2.edges), np.float32)
        ec.data.foreach_get("value", cur_e)
    ec.data.foreach_set("value", np.maximum(cur_e, border_edges.astype(np.float32)))
    vc = me2.attributes.get("crease_vert")
    if vc is None or vc.domain != 'POINT':
        vc = me2.attributes.new("crease_vert", 'FLOAT', 'POINT')
        cur_v = np.zeros(len(me2.vertices), np.float32)
    else:
        cur_v = np.empty(len(me2.vertices), np.float32)
        vc.data.foreach_get("value", cur_v)
    vc.data.foreach_set("value", np.maximum(cur_v, border_verts.astype(np.float32)))

    tmp_o = bpy.data.objects.new("NMTM_subd_tmp", me2)
    context.scene.collection.objects.link(tmp_o)
    try:
        mod = tmp_o.modifiers.new("NMTM_subd", 'SUBSURF')
        mod.subdivision_type = 'CATMULL_CLARK'
        mod.levels = level
        mod.render_levels = level
        mod.quality = 4                  # multires 默认
        mod.use_limit_surface = True
        mod.use_creases = True
        mod.uv_smooth = 'NONE'
        mod.boundary_smooth = 'ALL'
        dg = context.evaluated_depsgraph_get()
        out = bpy.data.meshes.new_from_object(tmp_o.evaluated_get(dg),
                                              preserve_all_data_layers=True, depsgraph=dg)
    finally:
        bpy.data.objects.remove(tmp_o, do_unlink=True)
        try:
            bpy.data.meshes.remove(me2)
        except Exception:
            pass
    return out


def _open_edge_segments(me, loop_uv):
    """低模开放边(单面边) → UV 线段端点对 (E,2,2), 供边缘衰减场。"""
    ecount = len(me.edges)
    if ecount == 0 or len(me.loops) == 0:
        return np.zeros((0, 2, 2), np.float32)
    le = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("edge_index", le)
    open_edge = np.bincount(le, minlength=ecount) == 1
    if not open_edge.any():
        return np.zeros((0, 2, 2), np.float32)
    ls = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_total", lt)
    nxt = np.arange(len(me.loops), dtype=np.int64) + 1
    nxt[ls + lt - 1] = ls                      # 面内环回: 末角的下一角是首角
    sel = np.flatnonzero(open_edge[le])
    return np.stack([loop_uv[sel], loop_uv[nxt[sel]]], axis=1).astype(np.float32)


def _get_island_labels(me, loop_vert, loop_uv, loop_total):
    """基面 → UV 岛标签(按网格内容指纹缓存)。"""
    fp = hash(loop_uv[:: max(1, loop_uv.shape[0] // 4096)].tobytes())
    key = (me.name_full, len(me.polygons), len(me.loops), fp)
    got = _island_cache.get(key)
    if got is None:
        poly_of_loop = np.repeat(np.arange(len(me.polygons), dtype=np.int64), loop_total)
        got = core.face_islands(loop_vert, loop_uv, poly_of_loop, len(me.polygons))
        _cache_put(_island_cache, key, got)
    return got


# ---------------------------------------------------------------------------
# 主构建
# ---------------------------------------------------------------------------

def _find_multires(obj):
    for m in obj.modifiers:
        if m.type == 'MULTIRES':
            return m
    return None


def _auto_level(corner_count, texel_count, fill, quad_budget):
    """最小 L 使 四边形数 = corners*4^(L-1) ≥ 有效texel数; 再按预算回退。"""
    needed = texel_count * fill
    level = 1
    while corner_count * (4 ** (level - 1)) < needed and level < MAX_LEVELS:
        level += 1
    while level > 1 and corner_count * (4 ** (level - 1)) > quad_budget:
        level -= 1
    return level


def build(context, obj, s, report):
    """核心构建。s = 场景设置 PropertyGroup。异常直接抛出, 由 Operator 兜底。"""
    t0 = time.perf_counter()
    me = obj.data

    if context.view_layer.objects.active is not obj:
        context.view_layer.objects.active = obj
    if obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    if me.uv_layers.active is None:
        raise RuntimeError("网格没有 UV 层, 法线贴图无从对应")

    # ---- 把无关物体临时摘出视图层求值: 每次 bpy.ops 都触发整场景深度图刷新,
    #      重型场景(如整只角色的骨骼变形网格)会被每个算子白算一遍 ----
    hidden_objs = []
    for o in context.view_layer.objects:
        if o.name != obj.name and not o.hide_get():
            try:
                o.hide_set(True)
                hidden_objs.append(o.name)
            except Exception:
                pass
    try:
        _build_inner(context, obj, s, report, t0)
    finally:
        vl_objects = context.view_layer.objects
        for name in hidden_objs:
            o = vl_objects.get(name)
            if o is not None:
                try:
                    o.hide_set(False)
                except Exception:
                    pass


def _build_inner(context, obj, s, report, t0):
    me = obj.data

    source = _resolve_source(obj, s)
    loop_uv = _read_loop_uvs(me)
    loop_vert = _read_loop_verts(me)
    # 工作分辨率 = 实际接入贴图的原生分辨率, 不再由用户猜数字(见 _native_resolution)
    bake_size = _native_resolution(obj, source, s.image if source == 'IMAGE' else None)
    print(f"[NormalMapToMesh] 工作分辨率(源贴图原生) = {bake_size}px")
    gx, gy, wmap = _gradients_cached(
        context, obj, me, source, s.image if source == 'IMAGE' else None,
        bake_size, loop_uv, loop_vert, bool(s.force_bake),
        s.deadzone_lsb / 127.5, s.slope_limit)

    if not (wmap > 0).any():
        raise RuntimeError("梯度全部无效(UV 未覆盖/法线异常), 高度重建失败")
    t_front = time.perf_counter()

    fill = _uv_fill(me, loop_uv)

    # 岛标签 + 岛界折痕集合(基面属性, 细分求值与岛处理共用)
    loop_total = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", loop_total)
    labels, n_islands = _get_island_labels(me, loop_vert, loop_uv, loop_total)
    border_edges, border_verts = _island_border_edges(me, loop_uv, loop_vert,
                                                     labels, loop_total)

    # ---- Multires 修改器 ----
    mod = _find_multires(obj)
    owned = bool(obj.get("nmtm_owned"))
    if mod is not None and mod.total_levels > 0 and not owned:
        raise RuntimeError(
            "物体已有带层级的 Multires(非本工具创建)。为防细节丢失请先应用或移除它。")
    if mod is None:
        mod = obj.modifiers.new("NormalMapToMesh", 'MULTIRES')
    if obj.modifiers.find(mod.name) != 0:
        bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=0)

    if s.auto_levels:
        level = _auto_level(len(me.loops), bake_size * bake_size, fill, s.quad_budget)
    else:
        level = min(s.levels, MAX_LEVELS)
        while level > 1 and len(me.loops) * (4 ** (level - 1)) > s.quad_budget:
            level -= 1
    level = max(1, level)
    quads = len(me.loops) * (4 ** (level - 1))

    # ---- Neumann(镜像)积分 (无风格化模糊) ----
    # 唯一的带限 = 顶点 Nyquist 抗混叠下限(采样理论要求, 非平滑参数):
    # 顶点间距粗于 texel 时, texel 级信号无法被网格表达, 不滤除即成混叠颗粒;
    # 自动级别(1 四边形/texel)下该下限为 0——完全无损
    texels_per_edge = float(np.sqrt(bake_size * bake_size * fill / max(quads, 1)))
    sigma_floor = 0.6 * texels_per_edge if texels_per_edge > 1.0 else 0.0
    field = core.integrate_height(
        gx, gy, smooth_sigma=(sigma_floor / bake_size) if sigma_floor > 0 else 0.0)

    # 开放边界(卡片边缘)衰减: 高度场沿 UV 距离场 smoothstep 归零。场量属于
    # 低模 UV 域, 与细分级别无关; 边界顶点另有硬锁零位移兜底
    fall_stat = ""
    if int(s.edge_falloff_px) > 0:
        seg = _open_edge_segments(me, loop_uv)
        if seg.shape[0]:
            field = field * core.edge_falloff_field(seg, bake_size,
                                                    int(s.edge_falloff_px))
            fall_stat = f" | 边缘衰减 {int(s.edge_falloff_px)}px/{seg.shape[0]:,}段"
    print(f"[NormalMapToMesh] 梯度有效率 {wmap.mean():.1%} | "
          f"高度场 p95 {np.percentile(np.abs(field[wmap > 0]), 95) * 1000:.2f}‰ | "
          f"抗混叠下限 σ {sigma_floor:.2f}px{fall_stat}")

    # ---- 建层(只为 Multires 数据结构; reshape 会完整覆写 MDISPS) ----
    # 层数已匹配则整段跳过(重建快路径); 建层在隐藏态跑——细分面从此只当
    # reshape 的容器, 目标面统一来自 Subsurf 求值, 与建层时的插值源无关
    if not (owned and mod.total_levels == level):
        mod.show_viewport = False
        try:
            if mod.total_levels > 0:
                mod.levels = 0
                mod.sculpt_levels = 0
                bpy.ops.object.multires_higher_levels_delete(modifier=mod.name)
            for _ in range(level):
                bpy.ops.object.multires_subdivide(modifier=mod.name, mode='CATMULL_CLARK')
        finally:
            mod.show_viewport = True
    mod.levels = level
    mod.sculpt_levels = level
    mod.render_levels = level
    if hasattr(mod, "uv_smooth"):
        mod.uv_smooth = 'NONE'          # 最终对象 UV 与采样时的线性插值一致
    t_subdiv = time.perf_counter()

    # ---- 细分基面: 岛界折痕锁定 CC 极限曲面(Subsurf 求值副本, 拓扑与 multires
    #      逐位一致) ----
    # 临时关掉其它修改器, 保证 reshape 空间纯净(骨架变形不得混入目标面)
    # 注意: bpy RNA 包装对象不能用 `is` 比较(每次访问都是新包装), 按类型过滤
    saved_vis = [(m, m.show_viewport) for m in obj.modifiers if m.type != 'MULTIRES']
    for m, _ in saved_vis:
        m.show_viewport = False
    tmp_obj = None
    tmp_me = None
    try:
        mod.show_viewport = False
        try:
            tmp_me = _subsurf_eval_mesh(context, obj, level, border_edges, border_verts)
        finally:
            mod.show_viewport = True
        t_eval = time.perf_counter()

        vcount = len(tmp_me.vertices)
        expected_loops = len(me.loops) * (4 ** (level - 1)) * 4
        if len(tmp_me.loops) != expected_loops:
            raise RuntimeError(
                f"Subsurf 细分拓扑异常: {len(tmp_me.loops):,} vs 预期 {expected_loops:,}")
        lv2 = _read_loop_verts(tmp_me)
        uv2 = _read_loop_uvs(tmp_me)

        # 逐 loop 采样(高度 + 有效权重)。三次 B 样条(C2): 位移曲面继承采样核的
        # 连续性——双线性的 C0 折面正是素模/雕刻视图"颗粒感"的来源
        samp = np.stack([field, wmap], axis=-1)
        s2 = core.sample_bspline_wrap(samp, uv2[:, 0], uv2[:, 1])
        h_loop = s2[:, 0].astype(np.float32)
        w_loop = (s2[:, 1] > 0.5).astype(np.float32)

        # 基面拓扑映射: 细分面按基面连续分块 → 逐 loop 岛标签
        per_face = loop_total.astype(np.int64) * (4 ** (level - 1))
        island_of_loop2 = np.repeat(np.repeat(labels, per_face), 4)
        if island_of_loop2.shape[0] != h_loop.shape[0]:
            raise RuntimeError(
                f"细分拓扑映射失配: {island_of_loop2.shape[0]:,} vs {h_loop.shape[0]:,}")

        # 位移方向 = 低模平滑角法线经 SIMPLE 细分线性插值的连续场(NMTM_N0 角
        # 属性), 插值后归一化——与 shader 逐像素插值法线同构, 面内 C∞;
        # 细分网格自身重算的离散角法线(SIMPLE 下逐基面跳变)禁止入场
        n0_attr = tmp_me.attributes.get("NMTM_N0")
        if n0_attr is None:
            raise RuntimeError("细分求值丢失 NMTM_N0 角法线属性(Subsurf 未插值 corner 属性?)")
        nbuf = np.empty(len(tmp_me.loops) * 3, np.float32)
        n0_attr.data.foreach_get("vector", nbuf)
        n0_loop = nbuf.reshape(-1, 3)
        n0_loop /= np.maximum(np.linalg.norm(n0_loop, axis=1), 1e-12)[:, None]
        h_loop = core.detrend_per_island(h_loop, uv2, island_of_loop2, n_islands, 'PLANE')
        h_loop = core.stitch_islands(h_loop, lv2, island_of_loop2, n_islands)
        h_loop *= w_loop   # 无效采样(未覆盖背景)不位移
        t_np1 = time.perf_counter()

        # 沿基准法线位移 × 高度倍数
        h_vert = core.average_loops_to_verts(h_loop, lv2, vcount)
        n0_vert = core.average_loop_vectors_to_verts(n0_loop * w_loop[:, None], lv2, vcount)
        ln = np.linalg.norm(n0_vert, axis=1)
        n0_vert /= np.maximum(ln, 1e-6)[:, None]
        amp_vert = h_vert * np.float32(s.disp_scale) * (ln > 0.1).astype(np.float32)
        dvec = n0_vert * amp_vert[:, None]

        # 边界硬锁: 开放边界顶点(卡片边缘)位移严格归零——边缘偏移会把原本
        # 贴合的卡片边撕出缝隙, 基面边缘本来就是对的; 平滑衰减带已在高度场
        # UV 域完成(edge_falloff_px, 细分不变), 此处只做精确兜底
        ecount = len(tmp_me.edges)
        ev = np.empty(ecount * 2, np.int32)
        tmp_me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        le = np.empty(len(tmp_me.loops), np.int32)
        tmp_me.loops.foreach_get("edge_index", le)
        edge_face_count = np.bincount(le, minlength=ecount)
        boundary_verts = np.unique(ev[edge_face_count[:ecount] == 1].ravel())
        if boundary_verts.size:
            dvec[boundary_verts] = 0.0
        t_np2 = time.perf_counter()

        co = _read_vert_cos(tmp_me)
        co += dvec
        tmp_me.vertices.foreach_set("co", co.ravel())
        tmp_me.update()
        mag = np.linalg.norm(dvec, axis=1)
        disp_stat = (f"{n_islands} 岛 | 边界锁定 {boundary_verts.size:,} 顶点 | "
                     f"位移幅值 p50 {np.percentile(mag, 50) * 1000:.2f} / "
                     f"p95 {np.percentile(mag, 95) * 1000:.2f} / "
                     f"max {mag.max() * 1000:.2f} (千分之一物体单位)")
        t_displace = time.perf_counter()
        print(f"[NormalMapToMesh] 位移明细: 建层 {t_subdiv - t_front:.1f}s"
              f" + 基面求值 {t_eval - t_subdiv:.1f}s"
              f" + 采样/岛处理 {t_np1 - t_eval:.1f}s + 锁边 {t_np2 - t_np1:.1f}s"
              f" + 写坐标 {t_displace - t_np2:.1f}s")

        # ---- reshape 写回 Multires 位移层 ----
        tmp_obj = bpy.data.objects.new("NMTM_reshape_tmp", tmp_me)
        context.scene.collection.objects.link(tmp_obj)
        tmp_obj.matrix_world = obj.matrix_world.copy()
        for o in list(context.selected_objects):
            o.select_set(False)
        tmp_obj.select_set(True)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.multires_reshape(modifier=mod.name)
    finally:
        if tmp_obj is not None:
            bpy.data.objects.remove(tmp_obj, do_unlink=True)
        if tmp_me is not None:
            try:
                bpy.data.meshes.remove(tmp_me)
            except Exception:
                pass
        for m, vis in saved_vis:
            m.show_viewport = vis

    # 高模细节按平滑着色观感正确
    obj.select_set(True)
    bpy.ops.object.shade_smooth()

    obj["nmtm_owned"] = 1
    obj["nmtm_level"] = level
    obj["nmtm_source"] = ("材质法线链" if source == 'MATERIAL'
                          else (s.image.name if s.image is not None else '?'))
    obj["nmtm_scale"] = float(s.disp_scale)

    t_end = time.perf_counter()
    msg = (f"{'材质' if source == 'MATERIAL' else '贴图'}求值 {bake_size}px | 级别 {level} | "
           f"{quads:,} 四边形 | {disp_stat} | "
           f"前端 {t_front - t0:.1f}s + 建层 {t_subdiv - t_front:.1f}s + "
           f"位移 {t_displace - t_subdiv:.1f}s + 写回 {t_end - t_displace:.1f}s "
           f"= {t_end - t0:.1f}s")
    print(f"[NormalMapToMesh] {obj.name}: {msg}")
    report({'INFO'}, msg)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _poll_mesh(context):
    obj = context.active_object
    return obj is not None and obj.type == 'MESH' and not obj.library


class NMTM_OT_build(bpy.types.Operator):
    """按面板设置构建/更新 Multires 细节(重复执行 = 从基面重建, 可反复调倍数)"""
    bl_idname = "nmtm.build"
    bl_label = "应用 / 更新"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _poll_mesh(context)

    def execute(self, context):
        s = context.scene.nmtm
        obj = context.active_object
        try:
            build(context, obj, s, self.report)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


class NMTM_OT_load_build(bpy.types.Operator, ImportHelper):
    """选择法线贴图文件, 加载后立即按贴图模式一键构建"""
    bl_idname = "nmtm.load_build"
    bl_label = "加载法线并一键构建"
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: bpy.props.StringProperty(
        default="*.png;*.jpg;*.jpeg;*.tga;*.tif;*.tiff;*.exr;*.bmp;*.webp;*.dds",
        options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _poll_mesh(context)

    def execute(self, context):
        s = context.scene.nmtm
        try:
            img = bpy.data.images.load(self.filepath, check_existing=True)
        except Exception as e:
            self.report({'ERROR'}, f"加载贴图失败: {e}")
            return {'CANCELLED'}
        s.image = img
        s.source = 'IMAGE'
        return bpy.ops.nmtm.build()


class NMTM_OT_remove(bpy.types.Operator):
    """移除本工具生成的 Multires 细节与修改器, 恢复低模"""
    bl_idname = "nmtm.remove"
    bl_label = "移除细节"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (_poll_mesh(context) and bool(obj.get("nmtm_owned"))
                and _find_multires(obj) is not None)

    def execute(self, context):
        obj = context.active_object
        if context.view_layer.objects.active is not obj:
            context.view_layer.objects.active = obj
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        mod = _find_multires(obj)
        if mod is not None:
            if mod.total_levels > 0:
                mod.levels = 0
                mod.sculpt_levels = 0
                bpy.ops.object.multires_higher_levels_delete(modifier=mod.name)
            bpy.ops.object.modifier_remove(modifier=mod.name)
        for k in ("nmtm_owned", "nmtm_level", "nmtm_image", "nmtm_source",
                  "nmtm_strength", "nmtm_scale"):
            if k in obj.keys():
                del obj[k]
        self.report({'INFO'}, "已恢复低模")
        return {'FINISHED'}


CLASSES = (NMTM_OT_build, NMTM_OT_load_build, NMTM_OT_remove)
