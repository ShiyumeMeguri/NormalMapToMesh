# -*- coding: utf-8 -*-
# NormalMapToMesh 操作层 —— bpy 侧: Multires 管理、depsgraph 求值、numpy 位移、reshape 写回。
#
# 流程(一键):
#   1. 读法线贴图 → FC 泊松积分出高度场(缓存, 换强度重跑时零成本)
#   2. 自动算细分级别(贴图texel数 × UV占用率 ≈ 四边形数, 受预算上限保护)
#   3. 建/重置 Multires → SIMPLE 细分 L 级(保形, 细节与烘焙面完全对位)
#   4. 关其它修改器 → depsgraph 求值出细分基面 → 逐 loop 采样高度 →
#      UV 岛去均值 → 逐顶点平均 → 沿顶点法线位移(强度 × texel世界尺度)
#   5. 临时物体 + multires_reshape 把绝对坐标写回 Multires 位移层 → 清理
#
# 重复点"应用"= 从基面重建(幂等), 改强度即所见即所得。

import time

import bpy
import numpy as np
from bpy_extras.io_utils import ImportHelper

from . import core

MAX_LEVELS = 9

# ---------------------------------------------------------------------------
# 运行期缓存(FFT 高度场 / UV 岛栅格), 换强度重跑时直接命中
# ---------------------------------------------------------------------------

_height_cache = {}
_island_cache = {}


def _cache_put(cache, key, val, cap=3):
    cache.pop(key, None)
    while len(cache) >= cap:
        cache.pop(next(iter(cache)))
    cache[key] = val


def clear_caches():
    _height_cache.clear()
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


def _read_vert_normals(me):
    n = len(me.vertices)
    buf = np.empty(n * 3, np.float32)
    me.vertex_normals.foreach_get("vector", buf)
    return buf.reshape(-1, 3)


def _read_image(img):
    """读像素为平铺 float32, 返回 (buf, w, h, ch, 指纹)。"""
    w, h = img.size
    if w == 0 or h == 0:
        raise RuntimeError(f"贴图 '{img.name}' 没有像素数据(文件缺失?)")
    ch = img.channels
    buf = np.empty(w * h * ch, np.float32)
    img.pixels.foreach_get(buf)
    fp = hash(buf[:: max(1, buf.size // 65536)].tobytes())
    return buf, w, h, ch, fp


def _detect_v_flip(buf, w, h, ch, sample_uv):
    """自动判定贴图内容相对网格 UV 是否上下颠倒(D3D 约定资产)。

    用采样点(loop UV + 三角形质心, 质心保证低模也有面内部样本)命中
    "有效法线像素"的比例投票: 正常 vs V翻转, 谁命中率高用谁。
    两者都极低说明贴图和网格根本不对应(照常返回 False)。
    """
    valid = core.valid_mask(buf, w, h, ch)
    ui = np.minimum((np.mod(sample_uv[:, 0], 1.0) * w).astype(np.int32), w - 1)
    vi = np.minimum((np.mod(sample_uv[:, 1], 1.0) * h).astype(np.int32), h - 1)
    hit_normal = float(valid[vi, ui].mean())
    hit_flip = float(valid[(h - 1) - vi, ui].mean())
    # 加性余量: 两者接近时不翻(乘性阈值在中等命中率下过脆)
    return hit_flip > hit_normal + 0.05 and hit_flip > 0.1, hit_normal, hit_flip


def _get_height_field(img_key, buf, w, h, ch, flip_green, reconstruct_z,
                      v_flip, highpass_px, smooth_px, bump_mode, deadzone_lsb,
                      slope_limit):
    """(缓存命中则跳过) 解码 + FC 积分 + 可选高通/平滑 + 凹凸方向判定。"""
    key = (img_key, w, h, ch, bool(flip_green), bool(reconstruct_z),
           bool(v_flip), float(highpass_px), float(smooth_px), str(bump_mode),
           float(deadzone_lsb), float(slope_limit))
    field = _height_cache.get(key)
    if field is None:
        gx, gy = core.decode_normals(buf, w, h, ch, flip_green, reconstruct_z, v_flip,
                                     deadzone=deadzone_lsb / 127.5)
        gx, gy = core.clamp_slope(gx, gy, slope_limit)
        hp_wavelength = (highpass_px / w) if highpass_px > 0 else 0.0
        smooth_sigma = (smooth_px / w) if smooth_px > 0 else 0.0
        field = core.integrate_height(gx, gy, hp_wavelength, smooth_sigma)
        if bump_mode == 'AUTO':
            # 不同生态凹凸约定相反(全局符号不可积判定), 用凸起先验的偏度判断
            valid = core.valid_mask(buf, w, h, ch)
            if v_flip:
                valid = valid[::-1]
            skew = core.height_skewness(field, valid)
            if skew < 0.0:
                field = -field
            print(f"[NormalMapToMesh] 凹凸判定: 偏度 {skew:+.3f}"
                  f" → {'反转(凹→凸)' if skew < 0 else '原样'}")
        elif bump_mode == 'INVERT':
            field = -field
        _cache_put(_height_cache, key, field)
    return field


def _base_tri_data(me, loop_vert, loop_uv, co):
    """低模三角化数据: texel 世界尺度 S、UV 占用率、V检测采样点。co=当前几何坐标。"""
    me.calc_loop_triangles()
    t = len(me.loop_triangles)
    tl = np.empty(t * 3, np.int32)
    me.loop_triangles.foreach_get("loops", tl)
    tp = np.empty(t, np.int32)
    me.loop_triangles.foreach_get("polygon_index", tp)

    tv = loop_vert[tl].reshape(-1, 3)
    p0, p1, p2 = co[tv[:, 0]], co[tv[:, 1]], co[tv[:, 2]]
    world_area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1).sum()

    uvt = loop_uv[tl].reshape(-1, 3, 2)
    e1 = uvt[:, 1] - uvt[:, 0]
    e2 = uvt[:, 2] - uvt[:, 0]
    uv_area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()

    texel_scale = float(np.sqrt(world_area / max(uv_area, 1e-12)))
    fill = float(min(max(uv_area, 0.02), 1.0))
    tri_centroid = uvt.mean(axis=1)   # V方向检测用的面内部采样点
    return texel_scale, fill, tri_centroid


def _eval_base_coords(context, obj, me):
    """"当前几何"基础坐标: 形态键等网格自身变形生效, 修改器(含 multires)全部排除。

    mesh.vertices.co 只是基础形状——带形态键的物体(如 UV 展平可视化网格)
    视口形状与基础形状完全不同, 背面判定/texel 尺度必须按当前几何算。
    """
    saved = [(m, m.show_viewport) for m in obj.modifiers]
    try:
        for m, _ in saved:
            m.show_viewport = False
        dg = context.evaluated_depsgraph_get()
        ev_me = obj.evaluated_get(dg).data
        if len(ev_me.vertices) == len(me.vertices):
            co = np.empty(len(ev_me.vertices) * 3, np.float32)
            ev_me.vertices.foreach_get("co", co)
            return co.reshape(-1, 3)
    finally:
        for m, vis in saved:
            m.show_viewport = vis
    return _read_vert_cos(me)


def _backface_signs(me, loop_vert, loop_uv, loop_total, coords):
    """双面卡片背面识别 → 逐基面位移符号 (+1/-1)。

    背面 = UV 绕向为负(与正面共享同一块 UV 但 loop 顺序反转) 且在"当前几何"
    上与某正绕向面共面反法线(BVH 最近点验证)。镜像复用 UV 的岛绕向也为负,
    但空间上不与正面重合, 不会被误判。背面位移取 -h 使卡片双面同向跟随
    正面形状; 卡缘顶点正负平均归零(正背面在此汇合, 物理正确)。
    coords 用求值后的基础坐标——形态键(如 UV 展平)会改变共面关系。
    """
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    fp = hash(loop_uv[:: max(1, loop_uv.shape[0] // 4096)].tobytes())
    cfp = hash(coords[:: max(1, coords.shape[0] // 4096)].tobytes())
    key = (me.name_full, len(me.polygons), len(me.loops), fp, cfp, 'bfsign')
    got = _island_cache.get(key)
    if got is not None:
        return got

    pcount = len(me.polygons)
    signs = np.ones(pcount, np.float32)
    # 逐面 UV 有向面积(loop_triangles 分块累加)
    me.calc_loop_triangles()
    t = len(me.loop_triangles)
    tl = np.empty(t * 3, np.int32)
    me.loop_triangles.foreach_get("loops", tl)
    tp = np.empty(t, np.int32)
    me.loop_triangles.foreach_get("polygon_index", tp)
    uvt = loop_uv[tl].reshape(-1, 3, 2)
    e1 = uvt[:, 1] - uvt[:, 0]
    e2 = uvt[:, 2] - uvt[:, 0]
    tri_signed = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    poly_uv_area = np.zeros(pcount, np.float64)
    np.add.at(poly_uv_area, tp, tri_signed)
    neg = np.nonzero(poly_uv_area < 0.0)[0]
    if neg.size == 0:
        _cache_put(_island_cache, key, signs)
        return signs

    # 当前几何上的面中心/法线/面积(全 numpy, 不读 me.polygons 的基础形状数据)
    tv = loop_vert[tl].reshape(-1, 3)
    p0, p1, p2 = coords[tv[:, 0]], coords[tv[:, 1]], coords[tv[:, 2]]
    tri_cross = np.cross(p1 - p0, p2 - p0)
    face_nrm = np.zeros((pcount, 3), np.float64)
    np.add.at(face_nrm, tp, tri_cross)
    face_area = 0.5 * np.linalg.norm(face_nrm, axis=1)
    face_nrm /= np.maximum(np.linalg.norm(face_nrm, axis=1, keepdims=True), 1e-20)
    csum = np.zeros((pcount, 3), np.float64)
    np.add.at(csum, np.repeat(np.arange(pcount), loop_total), coords[loop_vert])
    face_center = csum / loop_total[:, None]

    pos_mask = poly_uv_area >= 0.0
    verts = [tuple(v) for v in coords]
    polys = [p.vertices[:] for p, keep in zip(me.polygons, pos_mask) if keep]
    n_back = 0
    if polys:
        bvh = BVHTree.FromPolygons(verts, polys)
        for fi in neg:
            c = Vector(face_center[fi])
            hit = bvh.find_nearest(c)
            if hit[0] is None:
                continue
            tol = 0.5 * (float(face_area[fi]) ** 0.5) + 1e-9
            if (c - hit[0]).length < tol and hit[1].dot(Vector(face_nrm[fi])) < -0.5:
                signs[fi] = -1.0
                n_back += 1
    print(f"[NormalMapToMesh] 双面识别: 负绕向 {neg.size} 面, 判定背面 {n_back} 面")
    _cache_put(_island_cache, key, signs)
    return signs


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


def build(context, obj, img, s, report):
    """核心构建。s = 场景设置 PropertyGroup。异常直接抛出, 由 Operator 兜底。"""
    t0 = time.perf_counter()
    me = obj.data

    if context.view_layer.objects.active is not obj:
        context.view_layer.objects.active = obj
    if obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # 法线贴图按惯例应为 Non-Color(顺手修正, 不影响像素读取)。
    # 仅限文件图: 改色彩空间会触发生成型图像(如烘焙结果)重新生成, 清掉像素!
    try:
        if img.source == 'FILE' and img.colorspace_settings.name != 'Non-Color':
            img.colorspace_settings.name = 'Non-Color'
    except Exception:
        pass

    loop_vert = _read_loop_verts(me)
    loop_uv = _read_loop_uvs(me)
    base_coords = _eval_base_coords(context, obj, me)
    texel_scale, fill, tri_centroid = _base_tri_data(me, loop_vert, loop_uv, base_coords)

    buf, w, h, ch, img_fp = _read_image(img)
    if s.v_mode == 'AUTO':
        v_flip, hit_n, hit_f = _detect_v_flip(
            buf, w, h, ch, np.concatenate([loop_uv, tri_centroid]))
        print(f"[NormalMapToMesh] V方向检测: 正常命中 {hit_n:.0%} / 翻转命中 {hit_f:.0%}"
              f" → {'翻转' if v_flip else '正常'}")
        if max(hit_n, hit_f) < 0.1:
            report({'WARNING'}, "UV 落点几乎全是无效像素, 贴图可能与该网格不对应")
    else:
        v_flip = (s.v_mode == 'FLIP')

    if s.green_mode == 'AUTO':
        # 可积性判据: 真实高度场梯度无旋, 绿通道符号错误旋度显著增大
        gx0, gy0 = core.decode_normals(buf, w, h, ch, False, s.reconstruct_z, v_flip)
        r_gl = core.curl_residual(gx0, gy0)
        r_dx = core.curl_residual(gx0, -gy0)
        flip_green = r_dx < r_gl
        print(f"[NormalMapToMesh] 绿通道旋度: 不翻 {r_gl:.3e} / 翻 {r_dx:.3e}"
              f" → {'DirectX' if flip_green else 'OpenGL'} 约定")
        del gx0, gy0
    else:
        flip_green = (s.green_mode == 'DIRECTX')
    field = _get_height_field((img.name_full, img_fp), buf, w, h, ch,
                              flip_green, s.reconstruct_z, v_flip,
                              s.highpass_px, s.smooth_px, s.bump_mode, s.deadzone_lsb,
                              s.slope_limit)
    t_solve = time.perf_counter()

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
        level = _auto_level(len(me.loops), img.size[0] * img.size[1], fill, s.quad_budget)
    else:
        level = min(s.levels, MAX_LEVELS)
        while level > 1 and len(me.loops) * (4 ** (level - 1)) > s.quad_budget:
            level -= 1
    level = max(1, level)
    quads = len(me.loops) * (4 ** (level - 1))

    # ---- 重置到基面再细分(幂等: 重复应用/换强度不叠加) ----
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

        vcount = len(tmp_me.vertices)
        lv2 = _read_loop_verts(tmp_me)
        uv2 = _read_loop_uvs(tmp_me)

        # 逐 loop 采样高度(wrap 与 FFT 周期性一致)
        h_loop = core.sample_bilinear_wrap(field, uv2[:, 0], uv2[:, 1])

        # 细分面按基面连续分块(已实测验证), 拓扑映射精确且不怕 UV 岛重叠:
        # 基面 i (n 角) → n·4^(L-1) 个细分四边形 → 每面 4 个 loop
        loop_total = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("loop_total", loop_total)
        per_face = loop_total.astype(np.int64) * (4 ** (level - 1))

        def _expand(per_base_face_vals):
            out = np.repeat(np.repeat(per_base_face_vals, per_face), 4)
            if out.shape[0] != h_loop.shape[0]:
                raise RuntimeError(
                    f"细分拓扑映射失配: {out.shape[0]:,} vs {h_loop.shape[0]:,}")
            return out

        if s.detrend != 'OFF':
            labels, n_islands = _get_island_labels(me, loop_vert, loop_uv, loop_total)
            island_of_loop2 = _expand(labels)
            h_loop = core.detrend_per_island(h_loop, uv2, island_of_loop2,
                                             n_islands, s.detrend)
            # 岛间缝合: 消除长发/部件分段处的高度台阶
            h_loop = core.stitch_islands(h_loop, lv2, island_of_loop2, n_islands)

        if s.backface_fix:
            # 双面卡片: 背面位移取反, 使正背面同向跟随烘焙面形状
            signs = _backface_signs(me, loop_vert, loop_uv, loop_total, base_coords)
            if bool((signs < 0).any()):
                h_loop = h_loop * _expand(signs)

        h_vert = core.average_loops_to_verts(h_loop, lv2, vcount)

        co = _read_vert_cos(tmp_me)
        nrm = _read_vert_normals(tmp_me)
        co += nrm * (h_vert * np.float32(s.strength * texel_scale))[:, None]
        tmp_me.vertices.foreach_set("co", co.ravel())
        tmp_me.update()
        t_displace = time.perf_counter()

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
    obj["nmtm_image"] = img.name
    obj["nmtm_strength"] = float(s.strength)

    t_end = time.perf_counter()
    msg = (f"级别 {level} | {quads:,} 四边形 | "
           f"积分 {t_solve - t0:.1f}s + 细分 {t_subdiv - t_solve:.1f}s + "
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
        img = s.image
        if img is None:
            self.report({'ERROR'}, "请先选择法线贴图")
            return {'CANCELLED'}
        if obj.data.uv_layers.active is None:
            self.report({'ERROR'}, "网格没有 UV 层, 法线贴图无从对应")
            return {'CANCELLED'}
        try:
            build(context, obj, img, s, self.report)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


class NMTM_OT_load_build(bpy.types.Operator, ImportHelper):
    """选择法线贴图文件, 加载后立即一键构建(全自动)"""
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
        for k in ("nmtm_owned", "nmtm_level", "nmtm_image", "nmtm_strength"):
            if k in obj.keys():
                del obj[k]
        self.report({'INFO'}, "已恢复低模")
        return {'FINISHED'}


CLASSES = (NMTM_OT_build, NMTM_OT_load_build, NMTM_OT_remove)
