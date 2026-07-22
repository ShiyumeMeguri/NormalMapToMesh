# -*- coding: utf-8 -*-
# NormalMapToMesh 操作层 —— bpy 侧: Cycles 物体空间法线烘焙、Multires 管理、
# depsgraph 求值、numpy 向量位移、multires_reshape 写回。
#
# 流程(一键):
#   1. 在物体自身材质(或按所选贴图临时搭的简单节点链)上, 用 Cycles 烘焙
#      "物体空间法线"到临时浮点图——切线基/绿通道约定/通道重建/镜像 UV 岛
#      全部由渲染器按材质真实节点求值; 结果按指纹缓存(调强度重跑零成本)。
#   2. 自动算细分级别(烘焙分辨率 × UV 占用率 ≈ 四边形数, 受预算上限保护)。
#   3. 建/重置 Multires → SIMPLE 细分 L 级(保形)。
#   4. 关其它修改器 → depsgraph 求值细分基面 → 逐 loop 双线性采样烘焙图 →
#      Displace(RGB_TO_XYZ) 同款向量位移 → 逐顶点平均。
#   5. 临时物体 + multires_reshape 把绝对坐标写回 Multires 位移层 → 清理。
#
# 重复点"应用"= 从基面重建(幂等), 改强度即所见即所得。

import time

import bpy
import numpy as np
from bpy_extras.io_utils import ImportHelper

from . import core

MAX_LEVELS = 9
BAKE_MARGIN = 16
BAKE_SAMPLES = 16

# 运行期缓存: (网格指纹, 材质, 来源, 分辨率) -> 烘焙的 (H, W, 3) float32 像素。
# 注意只认网格/材质"身份", 不追踪材质节点内容变化——本会话内改了材质请重开会话
# 或换烘焙分辨率强制失效。
_bake_cache = {}


def _cache_put(cache, key, val, cap=2):
    cache.pop(key, None)
    while len(cache) >= cap:
        cache.pop(next(iter(cache)))
    cache[key] = val


def clear_caches():
    _bake_cache.clear()


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


def _bake_object_normals(context, obj, source, image, bake_size, loop_uv):
    """在物体材质上烘焙物体空间法线 → (H, W, 3) float32。按指纹缓存。

    烘焙期间临时: 切 Cycles(CPU)、隐藏其它物体、关掉本物体全部修改器的渲染求值
    (烘出静息态基面法线, 与写进 Multires 的位移空间一致)、往材质插烘焙目标节点。
    结束后全部还原。
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

    scene = context.scene
    saved_scene = (scene.render.engine, scene.cycles.device, scene.cycles.samples,
                   scene.cycles.bake_type, scene.render.bake.use_selected_to_active,
                   scene.render.bake.margin, scene.render.bake.normal_space)
    saved_hide = [(o.name, o.hide_render) for o in bpy.data.objects]
    saved_show_render = [(m.name, m.show_render) for m in obj.modifiers]
    saved_slots = None
    slot_appended = False
    tmp_mat = None
    bake_img = None
    inserted = []   # (材质名, 烘焙节点名, 原active节点名)
    try:
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = BAKE_SAMPLES
        scene.cycles.bake_type = 'NORMAL'
        scene.render.bake.use_selected_to_active = False
        scene.render.bake.margin = BAKE_MARGIN
        scene.render.bake.normal_space = 'OBJECT'

        for o in bpy.data.objects:
            o.hide_render = True
        obj.hide_render = False
        for o in list(context.selected_objects):
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        for m in obj.modifiers:
            m.show_render = False

        bake_img = bpy.data.images.new("NMTM_bake", width=bake_size, height=bake_size,
                                       float_buffer=True)
        bake_img.colorspace_settings.name = 'Non-Color'

        if source == 'IMAGE':
            # 法线贴图按惯例应为 Non-Color(仅文件图: 改生成型图的色彩空间会清掉像素)
            try:
                if image.source == 'FILE' and image.colorspace_settings.name != 'Non-Color':
                    image.colorspace_settings.name = 'Non-Color'
            except Exception:
                pass
            tmp_mat = bpy.data.materials.new("NMTM_bake_mat")
            nt = tmp_mat.node_tree
            nt.nodes.clear()
            out = nt.nodes.new('ShaderNodeOutputMaterial')
            bsdf = nt.nodes.new('ShaderNodeBsdfDiffuse')
            nmap = nt.nodes.new('ShaderNodeNormalMap')
            timg = nt.nodes.new('ShaderNodeTexImage')
            timg.image = image
            nt.links.new(nmap.inputs['Color'], timg.outputs['Color'])
            nt.links.new(bsdf.inputs['Normal'], nmap.outputs['Normal'])
            nt.links.new(out.inputs['Surface'], bsdf.outputs['BSDF'])
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

        bpy.ops.object.bake(type='NORMAL')

        buf = np.empty(bake_size * bake_size * 4, np.float32)
        bake_img.pixels.foreach_get(buf)
        rgb = buf.reshape(bake_size, bake_size, 4)[..., :3].astype(np.float32, copy=True)
        if float(rgb.std()) < 1e-5:
            raise RuntimeError("烘焙结果是纯色, 物体空间法线烘焙未生效(检查材质与 UV)")
    finally:
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

    _cache_put(_bake_cache, key, rgb)
    return rgb


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

    source = _resolve_source(obj, s)
    loop_uv = _read_loop_uvs(me)
    bake_size = int(s.bake_size)
    rgb = _bake_object_normals(context, obj, source,
                               s.image if source == 'IMAGE' else None,
                               bake_size, loop_uv)
    t_bake = time.perf_counter()

    fill = _uv_fill(me, loop_uv)

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

        # 逐 loop 采样烘焙图 → Displace(RGB_TO_XYZ) 同款物体空间向量位移
        rgb_loop = core.sample_bilinear_wrap(rgb, uv2[:, 0], uv2[:, 1])
        offset_loop = core.displace_rgb_to_xyz(rgb_loop, s.disp_strength)
        offset_vert = core.average_loop_vectors_to_verts(offset_loop, lv2, vcount)

        co = _read_vert_cos(tmp_me)
        co += offset_vert
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
    obj["nmtm_source"] = ("材质法线链" if source == 'MATERIAL'
                          else (s.image.name if s.image is not None else '?'))
    obj["nmtm_strength"] = float(s.disp_strength)

    t_end = time.perf_counter()
    msg = (f"{'材质' if source == 'MATERIAL' else '贴图'}烘焙 {bake_size}px | 级别 {level} | "
           f"{quads:,} 四边形 | 烘焙 {t_bake - t0:.1f}s + 细分 {t_subdiv - t_bake:.1f}s + "
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
        for k in ("nmtm_owned", "nmtm_level", "nmtm_image", "nmtm_source", "nmtm_strength"):
            if k in obj.keys():
                del obj[k]
        self.report({'INFO'}, "已恢复低模")
        return {'FINISHED'}


CLASSES = (NMTM_OT_build, NMTM_OT_load_build, NMTM_OT_remove)
