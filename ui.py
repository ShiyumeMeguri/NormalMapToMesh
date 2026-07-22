# -*- coding: utf-8 -*-
# NormalMapToMesh UI —— 3D视图侧栏面板 + 场景设置。

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty)


class NMTMSettings(bpy.types.PropertyGroup):
    source: EnumProperty(
        name="法线来源",
        items=(('AUTO', "自动",
                "材质里有 Normal Map 节点就烘焙材质(最忠实, 含游戏资产的通道重建网络); "
                "否则用下面选择的贴图"),
               ('MATERIAL', "材质",
                "烘焙物体现有材质的法线输出——绿通道约定/通道重建由材质节点自己保证"),
               ('IMAGE', "贴图",
                "忽略材质, 用所选贴图按标准切线空间解释临时搭节点链烘焙")),
        default='AUTO')
    image: PointerProperty(
        type=bpy.types.Image, name="法线贴图",
        description="贴图模式使用: 与当前网格 UV 对应的切线空间法线贴图")
    disp_scale: FloatProperty(
        name="高度倍数", default=1.0, soft_min=-3.0, soft_max=3.0, step=10, precision=2,
        description="1.0 = 按法线坡度积分出的物理高度(物体空间单位, 高频细节自动"
                    "获得与波长匹配的小高度)。完全平贴严格零位移(无整体膨胀), "
                    "凹凸随倾斜方向正负; 负值整体反向")
    bake_size: EnumProperty(
        name="烘焙分辨率",
        items=(('512', "512", ""), ('1024', "1024", ""),
               ('2048', "2048", ""), ('4096', "4096", "")),
        default='2048',
        description="物体空间法线烘焙图分辨率, 通常取源法线贴图同档")
    edge_smooth_iters: IntProperty(
        name="边缘/位移平滑", default=8, min=0, max=64,
        description="位移向量场的图拉普拉斯平滑迭代次数: 开放边界(卡片边缘)顶点"
                    "始终锁死为零位移(防边缘错位撕缝), 平滑让位移向边缘平滑衰减"
                    "并去除高频斑点; 0=只锁边不平滑")
    force_bake: BoolProperty(
        name="强制烘焙路径", default=False,
        description="跳过直算前端, 强制走 Cycles EMIT 烘焙(兼容任意材质节点网络; "
                    "默认关——直算不支持的节点会自动回退烘焙)")
    deadzone_lsb: FloatProperty(
        name="平整死区 (LSB)", default=1.0, min=0.0, soft_max=4.0, step=10, precision=1,
        description="法线 XY 分量低于该值(8bit 台阶数)视为纯平——量化噪声经积分会"
                    "放大成低频起伏, 死区保证平坦区严格为平(0=关)")
    slope_limit: FloatProperty(
        name="坡度上限", default=10.0, min=0.0, soft_max=20.0, step=10, precision=1,
        description="坡度(tanθ)限幅, 压制烘焙噪声/压缩伪影导致的尖刺(0=关)")
    subdiv_mode: EnumProperty(
        name="细分方式",
        items=(('SIMPLE', "Simple (保形)", "保持低模形状, 细节与烘焙面完全对位"),
               ('CATMULL_CLARK', "平滑 (Catmull-Clark)",
                "平滑基面; 开放边界的收缩漂移会用 SIMPLE 对照细分自动校正回基面边缘"
                "(边缘钉死不产生缝隙), 首次构建多一次对照细分开销"),
               ('LINEAR', "Linear", "线性细分")),
        default='SIMPLE')
    auto_levels: BoolProperty(
        name="自动级别", default=True,
        description="按烘焙分辨率 × UV 占用率自动匹配 ≈1 四边形/texel")
    levels: IntProperty(
        name="级别", default=6, min=1, max=9,
        description="手动指定 Multires 细分级别")
    quad_budget: IntProperty(
        name="四边形预算", default=16_000_000, min=10_000, max=120_000_000,
        description="细分产生的四边形上限(内存保护); 自动与手动级别都受此约束")


class NMTM_PT_panel(bpy.types.Panel):
    bl_label = "法线 → 多级精度高模"
    bl_idname = "NMTM_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Ruri"

    def draw(self, context):
        layout = self.layout
        s = context.scene.nmtm
        obj = context.active_object

        col = layout.column()
        col.scale_y = 1.4
        col.operator("nmtm.load_build", icon='FILE_IMAGE')

        layout.prop(s, "source", text="来源")
        if s.source != 'MATERIAL':
            layout.template_ID(s, "image", open="image.open")
        layout.prop(s, "disp_scale", slider=True)
        row = layout.row()
        row.scale_y = 1.2
        row.operator("nmtm.build", icon='MOD_MULTIRES')

        box = layout.box()
        box.label(text="选项", icon='PREFERENCES')
        box.prop(s, "bake_size", text="分辨率")
        box.prop(s, "edge_smooth_iters")
        box.prop(s, "deadzone_lsb")
        box.prop(s, "slope_limit")
        box.prop(s, "subdiv_mode", text="细分")
        row = box.row()
        row.prop(s, "auto_levels")
        sub = row.row()
        sub.active = not s.auto_levels
        sub.prop(s, "levels", text="")
        box.prop(s, "quad_budget")
        box.prop(s, "force_bake")

        if obj is not None and obj.get("nmtm_owned"):
            box = layout.box()
            box.label(text="当前状态", icon='CHECKMARK')
            box.label(text=f"来源: {obj.get('nmtm_source', obj.get('nmtm_image', '?'))}")
            box.label(text=f"级别 {obj.get('nmtm_level', '?')} | "
                           f"倍数 {obj.get('nmtm_scale', 0.0):.2f}")
            box.operator("nmtm.remove", icon='TRASH')


CLASSES = (NMTMSettings, NMTM_PT_panel)


def register_props():
    bpy.types.Scene.nmtm = PointerProperty(type=NMTMSettings)


def unregister_props():
    del bpy.types.Scene.nmtm
