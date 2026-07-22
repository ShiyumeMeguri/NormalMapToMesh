# -*- coding: utf-8 -*-
# NormalMapToMesh 操作层 —— bpy 侧: Cycles 物体空间法线烘焙、Multires 管理、
# depsgraph 求值、numpy 向量位移、multires_reshape 写回。
#
# 流程(一键):
#   1. 用 Cycles EMIT 烘焙三张物体空间图(按指纹缓存): 带法线贴图的扰动法线 n1
#      (材质 Normal Map 节点输出 / 所选贴图临时链)、纯平滑基准法线 n0
#      (Geometry.Normal)、表面位置 P(Geometry.Position, 原始浮点)。
#      切线基/绿通道约定/通道重建/镜像 UV 岛全部由渲染器按材质真实节点求值。
#   2. 零猜测装配高度梯度 dh/du = (n0 − n1/(n1·n0))·∂P/∂u → FC 频域泊松积分
#      → 物理高度场(物体单位); 平贴处梯度严格 0 → 高度 0, 无整体膨胀。
#   3. 自动算细分级别 → 建/重置 Multires → SIMPLE 细分 L 级(保形)。
#   4. 关其它修改器 → depsgraph 求值细分基面 → 逐 loop 采样高度 → 逐岛去趋势
#      → 岛间缝合 → 逐顶点平均 → 沿基准法线位移 × 高度倍数。
#   5. 临时物体 + multires_reshape 把绝对坐标写回 Multires 位移层 → 清理。
#
# 重复点"应用"= 从基面重建(幂等), 改倍数即所见即所得。

import time

import bpy
import numpy as np
from bpy_extras.io_utils import ImportHelper

from . import core

MAX_LEVELS = 9
BAKE_MARGIN = 16
BAKE_SAMPLES = 1   # EMIT 烘焙无噪声, 1 采样足够

# 运行期缓存: (网格指纹, 材质, 来源, 分辨率) -> 三张烘焙图 (n1, n0, P)。
# 注意只认网格/材质"身份", 不追踪材质节点内容变化——本会话内改了材质请重开会话
# 或换烘焙分辨率强制失效。
_bake_cache = {}
_island_cache = {}


def _cache_put(cache, key, val, cap=2):
    cache.pop(key, None)
    while len(cache) >= cap:
        cache.pop(next(iter(cache)))
    cache[key] = val


def clear_caches():
    _bake_cache.clear()
    _island_cache.clear()
    _simple_cache.clear()


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
            raise RuntimeError("物体材质里没有 Normal Map 节点, 无法按材质烘焙; 请改用贴图模式")
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


# ---------------------------------------------------------------------------
# Cycles 物体空间法线烘焙
# ---------------------------------------------------------------------------

def _mesh_fingerprint(me, loop_uv):
    fp = hash(loop_uv[:: max(1, loop_uv.shape[0] // 4096)].tobytes())
    return (me.name_full, len(me.vertices), len(me.polygons), len(me.loops), fp)


def _build_encoder(nt, src_socket, vector_type='NORMAL', encode=True):
    """向量(世界空间) → Vector Transform 转物体空间 → (可选 ×0.5+0.5 编码) → Emission。

    法线用 NORMAL 类型+编码; 位置用 POINT 类型+原始浮点(float 图保 HDR)。
    返回 (新建节点列表, emission 节点)。调用方负责把 emission 接到材质输出并清理。
    """
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
    """单次物体空间 EMIT 烘焙 → (H, W, 3) float32。

    Blender 5.x 的 Cycles NORMAL 烘焙只输出几何平滑法线, **不含**材质法线贴图
    扰动(实测两图逐位相同), 因此用 EMIT 发射色编码烘出所需向量。
    kind='DETAIL': 扰动法线 n1——MATERIAL 来源把每个材质的 Normal Map 节点输出
                   临时接进编码链(无 Normal Map 的材质接 Geometry.Normal → 零梯度);
                   IMAGE 来源用所选贴图临时搭 TexImage→NormalMap→编码链。
    kind='BASELINE': 基准法线 n0——全部槽临时换 Geometry.Normal 编码链。
    kind='POSITION': 表面位置 P——全部槽临时换 Geometry.Position 原始浮点链。
    场景/选中/隐藏/修改器状态由 _bake_triple 负责; 本函数管材质/节点/目标图, 结束全还原。
    """
    me = obj.data
    bake_img = None
    tmp_mat = None
    saved_slots = None
    slot_appended = False
    inserted = []   # (材质名, 烘焙目标节点名, 原active节点名)
    grafts = []     # (材质名, [临时节点名], 输出节点名, (原surface来源节点名, 输出口名)|None)
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
    """三次同参数物体空间 EMIT 烘焙(n1, n0, P) → 三张 (H,W,3)。按指纹缓存。

    烘焙期间临时: 切 Cycles(CPU)、隐藏其它物体、关掉本物体全部修改器的渲染求值
    (烘出静息态基面数据, 与写进 Multires 的位移空间一致)。结束后全部还原。
    """
    me = obj.data
    mats = tuple(s.material.name_full if s.material else '' for s in obj.material_slots)
    key = (_mesh_fingerprint(me, loop_uv), mats, source,
           image.name_full if image is not None else '', int(bake_size))
    got = _bake_cache.get(key)
    if got is not None:
        return got

    if source == 'MATERIAL':
        for slot in obj.material_slots:
            if slot.material is not None and slot.material.library is not None:
                raise RuntimeError(
                    f"材质 '{slot.material.name}' 来自链接库, 无法插入烘焙节点; 请先 Make Local")
    if source == 'IMAGE':
        # 法线贴图按惯例应为 Non-Color(仅文件图: 改生成型图的色彩空间会清掉像素)
        try:
            if image.source == 'FILE' and image.colorspace_settings.name != 'Non-Color':
                image.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass

    scene = context.scene
    saved_scene = (scene.render.engine, scene.cycles.device, scene.cycles.samples,
                   scene.cycles.bake_type, scene.render.bake.use_selected_to_active,
                   scene.render.bake.margin, scene.render.bake.normal_space)
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

        tb0 = time.perf_counter()
        rgb_detail = _bake_once(context, obj, 'DETAIL', source, image, bake_size)
        tb1 = time.perf_counter()
        rgb_base = _bake_once(context, obj, 'BASELINE', source, None, bake_size)
        tb2 = time.perf_counter()
        pos_map = _bake_once(context, obj, 'POSITION', source, None, bake_size)
        tb3 = time.perf_counter()
        print(f"[NormalMapToMesh] 烘焙明细: n1 {tb1 - tb0:.1f}s + n0 {tb2 - tb1:.1f}s"
              f" + P {tb3 - tb2:.1f}s")
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
         scene.render.bake.margin, scene.render.bake.normal_space) = saved_scene

    triple = (rgb_detail, rgb_base, pos_map)
    _cache_put(_bake_cache, key, triple)
    return triple


_simple_cache = {}


def _eval_simple_coords(context, obj, level):
    """在网格副本上做同级 SIMPLE 多级细分求值 → 顶点坐标 (V,3)。按指纹缓存。

    Catmull-Clark(平滑)细分会把开放边界向内收缩——锁位移锁不住细分自身的漂移。
    SIMPLE 细分与 CC 细分的拓扑/顶点序完全一致(同一均匀细化, 模式只影响坐标),
    其边界顶点恰好留在基面边缘原位, 用作边界校正的目标位置。
    副本不带任何其它修改器(与主求值隐藏 Armature 的口径一致)。
    """
    me = obj.data
    key = (me.name_full, len(me.vertices), len(me.loops), int(level), 'simple')
    got = _simple_cache.get(key)
    if got is not None:
        return got

    tmp_o = None
    me2 = None
    try:
        me2 = me.copy()
        tmp_o = bpy.data.objects.new("NMTM_simple_tmp", me2)
        context.scene.collection.objects.link(tmp_o)
        for o in list(context.selected_objects):
            o.select_set(False)
        tmp_o.select_set(True)
        context.view_layer.objects.active = tmp_o
        mod = tmp_o.modifiers.new("NMTM_simple", 'MULTIRES')
        # 关键: multires 层数据(MDISPS)存在 Mesh 数据块里, me.copy() 会带过来,
        # 新建修改器直接认领——必须先清空, 否则"SIMPLE 对照"输出的是旧 CC/位移面,
        # 边界校正沦为空转(v3.2 的隐性 bug; 重建时甚至会带上上次的真实位移)。
        if mod.total_levels > 0:
            mod.levels = 0
            mod.sculpt_levels = 0
            bpy.ops.object.multires_higher_levels_delete(modifier=mod.name)
        # 注意: 不能隐藏状态下 subdivide(细分算子以当前求值面为插值源, 会改变结果字节)
        for _ in range(level):
            bpy.ops.object.multires_subdivide(modifier=mod.name, mode='SIMPLE')
        mod.levels = level
        dg = context.evaluated_depsgraph_get()
        ev_me = tmp_o.evaluated_get(dg).data
        n = len(ev_me.vertices)
        buf = np.empty(n * 3, np.float32)
        ev_me.vertices.foreach_get("co", buf)
        coords = buf.reshape(-1, 3).copy()
    finally:
        if tmp_o is not None:
            bpy.data.objects.remove(tmp_o, do_unlink=True)
        if me2 is not None:
            try:
                bpy.data.meshes.remove(me2)
            except Exception:
                pass
    _cache_put(_simple_cache, key, coords)
    return coords


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
    #      重型场景(如整只角色的骨骼变形网格)会被每个算子白算一遍——它们不参与
    #      本管线任何数据(烘焙走 hide_render 独立管理; obj 自己的骨架物体即使
    #      隐藏也仍作为依赖被求值), 纯环境开销, 结束后全部还原 ----
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
    bake_size = int(s.bake_size)
    rgb_detail, rgb_base, pos_map = _bake_triple(context, obj, source,
                                                 s.image if source == 'IMAGE' else None,
                                                 bake_size, loop_uv)

    # 零猜测高度重建: 梯度 = (n0 − n1/(n1·n0))·∂P/∂{u,v} → 频域泊松积分(物理单位)
    gx, gy, wmap = core.height_gradients(rgb_detail, rgb_base, pos_map)
    field = core.integrate_height(gx, gy)
    if not (wmap > 0).any():
        raise RuntimeError("烘焙图全部无效(法线长度异常), 高度重建失败")
    print(f"[NormalMapToMesh] 梯度有效率 {wmap.mean():.1%} | "
          f"高度场 p95 {np.percentile(np.abs(field[wmap > 0]), 95) * 1000:.2f}‰")
    t_bake = time.perf_counter()

    fill = _uv_fill(me, loop_uv)

    # ---- Multires 修改器 ----
    mod = _find_multires(obj)
    pre_mod = mod   # 提前对照细分时若已有旧层, 先按住它的求值(数据不动, 纯环境开销)
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

    # ---- CC 边界校正的对照细分提前算(此刻场景还没有任何重型 multires):
    #      对照细分是 (基面网格, 级别) 的纯函数, 调用时机不影响其输出 ----
    t_presub = time.perf_counter()
    coords_simple = None
    if s.subdiv_mode == 'CATMULL_CLARK':
        if pre_mod is not None and pre_mod.total_levels > 0:
            # 重建场景: 旧层还在, 按住其视口求值(数据不动, 纯环境开销)
            pre_mod.show_viewport = False
        try:
            coords_simple = _eval_simple_coords(context, obj, level)
        finally:
            if pre_mod is not None and pre_mod.total_levels > 0:
                pre_mod.show_viewport = True
        context.view_layer.objects.active = obj
    t_simple = time.perf_counter()

    # ---- 重置到基面再细分(幂等: 重复应用/换强度不叠加) ----
    # 注意: 不能在修改器隐藏状态下跑 subdivide——细分算子以当前求值面为插值源,
    # 隐藏会改变 MDISPS 初始化(实测字节级不等), 只能吃下逐级重求值的开销
    if mod.total_levels > 0:
        mod.levels = 0
        mod.sculpt_levels = 0
        bpy.ops.object.multires_higher_levels_delete(modifier=mod.name)
    for _ in range(level):
        bpy.ops.object.multires_subdivide(modifier=mod.name, mode=s.subdiv_mode)
    mod.levels = level
    mod.sculpt_levels = level
    mod.render_levels = level
    t_subdiv = time.perf_counter()

    # ---- 求值细分基面(临时关掉其它修改器, 保证 reshape 空间纯净) ----
    # 注意: bpy RNA 包装对象不能用 `is` 比较(每次访问都是新包装), 按类型过滤
    saved_vis = [(m, m.show_viewport) for m in obj.modifiers if m.type != 'MULTIRES']
    for m, _ in saved_vis:
        m.show_viewport = False
    tmp_obj = None
    tmp_me = None
    try:
        dg = context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        tmp_me = bpy.data.meshes.new_from_object(
            ev, preserve_all_data_layers=True, depsgraph=dg)
        t_eval = time.perf_counter()

        vcount = len(tmp_me.vertices)
        lv2 = _read_loop_verts(tmp_me)
        uv2 = _read_loop_uvs(tmp_me)

        # 逐 loop 采样(高度 + 基准法线 + 有效权重, 拼一次采样)
        samp = np.concatenate([field[..., None], rgb_base, wmap[..., None]], axis=2)
        s5 = core.sample_bilinear_wrap(samp, uv2[:, 0], uv2[:, 1])
        h_loop = s5[:, 0].astype(np.float32)
        n0_loop, w0_loop = core.decode_unit_normal(s5[:, 1:4])
        w_loop = (s5[:, 4] > 0.5).astype(np.float32) * w0_loop

        # 逐岛去趋势+缝合: 全局积分给每岛留下的任意偏移/斜坡, 用基面拓扑映射
        # (细分面按基面连续分块)精确归岛后消除
        loop_total = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("loop_total", loop_total)
        labels, n_islands = _get_island_labels(me, loop_vert, loop_uv, loop_total)
        per_face = loop_total.astype(np.int64) * (4 ** (level - 1))
        island_of_loop2 = np.repeat(np.repeat(labels, per_face), 4)
        if island_of_loop2.shape[0] != h_loop.shape[0]:
            raise RuntimeError(
                f"细分拓扑映射失配: {island_of_loop2.shape[0]:,} vs {h_loop.shape[0]:,}")
        h_loop = core.detrend_per_island(h_loop, uv2, island_of_loop2, n_islands, 'PLANE')
        h_loop = core.stitch_islands(h_loop, lv2, island_of_loop2, n_islands)
        h_loop *= w_loop   # 无效采样(未烘焙背景)不位移
        t_np1 = time.perf_counter()

        # 沿基准法线位移 × 高度倍数
        h_vert = core.average_loops_to_verts(h_loop, lv2, vcount)
        n0_vert = core.average_loop_vectors_to_verts(n0_loop * w_loop[:, None], lv2, vcount)
        ln = np.linalg.norm(n0_vert, axis=1)
        n0_vert /= np.maximum(ln, 1e-6)[:, None]
        amp_vert = h_vert * np.float32(s.disp_scale) * (ln > 0.1).astype(np.float32)
        dvec = n0_vert * amp_vert[:, None]

        # 边缘锁定 + 位移场平滑(等价: 多级精度平滑后沿边缘刷"擦除多级精度置换"):
        # 开放边界顶点(卡片边缘)位移严格归零——边缘偏移会把原本贴合的卡片边
        # 撕出缝隙, 基面边缘本来就是对的; 再对位移向量场做图拉普拉斯平滑
        # (边界 Dirichlet 0), 位移向边缘平滑衰减, 同时去除高频斑点。
        ecount = len(tmp_me.edges)
        ev = np.empty(ecount * 2, np.int32)
        tmp_me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        le = np.empty(len(tmp_me.loops), np.int32)
        tmp_me.loops.foreach_get("edge_index", le)
        edge_face_count = np.bincount(le, minlength=ecount)
        boundary_verts = np.unique(ev[edge_face_count[:ecount] == 1].ravel())
        e0 = ev[:, 0].astype(np.int64)
        e1 = ev[:, 1].astype(np.int64)
        if boundary_verts.size:
            dvec[boundary_verts] = 0.0
        iters = int(s.edge_smooth_iters)
        if iters > 0 and ecount:
            deg = (np.bincount(e0, minlength=vcount)
                   + np.bincount(e1, minlength=vcount)).astype(np.float64)
            deg = np.maximum(deg, 1.0)
            for _ in range(iters):
                nb = np.empty_like(dvec)
                for c in range(3):
                    sc = (np.bincount(e0, weights=dvec[e1, c], minlength=vcount)
                          + np.bincount(e1, weights=dvec[e0, c], minlength=vcount))
                    nb[:, c] = (sc / deg).astype(np.float32)
                dvec = 0.5 * dvec + 0.5 * nb
                if boundary_verts.size:
                    dvec[boundary_verts] = 0.0

        co = _read_vert_cos(tmp_me)

        # Catmull-Clark(平滑)细分的边界校正: CC 把开放边界向内收缩, 锁位移锁不住
        # 细分自身的漂移——用同拓扑 SIMPLE 对照细分把边界顶点拉回基面边缘原位,
        # 校正量沿图距离在 K 环内线性衰减, 平滑融入 CC 内部
        t_np2 = time.perf_counter()
        corr_stat = ""
        if coords_simple is not None and boundary_verts.size and ecount:
            if coords_simple.shape[0] != vcount:
                print(f"[NormalMapToMesh] 警告: SIMPLE 对照细分拓扑不匹配"
                      f"({coords_simple.shape[0]:,} vs {vcount:,}), 跳过边界校正")
            else:
                k_rings = max(2, 2 ** (level - 1))
                dist = np.full(vcount, k_rings + 1, np.int32)
                dist[boundary_verts] = 0
                for _ in range(k_rings):
                    np.minimum.at(dist, e1, dist[e0] + 1)
                    np.minimum.at(dist, e0, dist[e1] + 1)
                wgt = np.clip(1.0 - dist.astype(np.float32) / k_rings, 0.0, 1.0)
                corr = (coords_simple - co) * wgt[:, None]
                co += corr
                corr_stat = (f" | CC边界校正 max "
                             f"{np.linalg.norm(corr, axis=1).max() * 1000:.2f}‰/{k_rings}环")

        t_ccfix = time.perf_counter()
        co += dvec
        tmp_me.vertices.foreach_set("co", co.ravel())
        tmp_me.update()
        mag = np.linalg.norm(dvec, axis=1)
        disp_stat = (f"{n_islands} 岛 | 边界锁定 {boundary_verts.size:,} 顶点{corr_stat} | "
                     f"位移幅值 p50 {np.percentile(mag, 50) * 1000:.2f} / "
                     f"p95 {np.percentile(mag, 95) * 1000:.2f} / "
                     f"max {mag.max() * 1000:.2f} (千分之一物体单位)")
        t_displace = time.perf_counter()
        print(f"[NormalMapToMesh] 位移明细: 对照细分 {t_simple - t_presub:.1f}s"
              f" + 求值拷贝 {t_eval - t_subdiv:.1f}s"
              f" + 采样/岛处理 {t_np1 - t_eval:.1f}s + 锁边/平滑 {t_np2 - t_np1:.1f}s"
              f" + CC校正 {t_ccfix - t_np2:.1f}s + 写坐标 {t_displace - t_ccfix:.1f}s")

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
    msg = (f"{'材质' if source == 'MATERIAL' else '贴图'}烘焙 {bake_size}px | 级别 {level} | "
           f"{quads:,} 四边形 | {disp_stat} | "
           f"烘焙 {t_bake - t0:.1f}s + 细分 {t_subdiv - t_bake:.1f}s + "
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
    """按面板设置构建/更新 Multires 细节(重复执行 = 从基面重建, 可反复调强度)"""
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
