# -*- coding: utf-8 -*-
# NormalMapToMesh UI —— 3D视图侧栏面板 + 场景设置。

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty)


class NMTMSettings(bpy.types.PropertyGroup):
    image: PointerProperty(
        type=bpy.types.Image, name="法线贴图",
        description="与当前选中网格 UV 完全对应的切线空间法线贴图")
    strength: FloatProperty(
        name="强度", default=1.0, soft_min=-5.0, soft_max=5.0, step=5,
        description="位移强度缩放。1.0 = 按贴图 texel 密度推算的物理高度; 负值反向")
    bump_mode: EnumProperty(
        name="凹凸方向",
        items=(('AUTO', "自动 (凸起优先)",
                "按高度分布偏度判定全局凹凸符号——不同工具链约定相反, 细节以凸起为主时自动纠正"),
               ('RAW', "原样", "按标准切线空间数学直接积分"),
               ('INVERT', "反转", "整体高度取负(凹↔凸)")),
        default='AUTO')
    green_mode: EnumProperty(
        name="绿通道",
        items=(('AUTO', "自动", "用可积性(旋度)判据自动判定 OpenGL/DirectX 约定"),
               ('OPENGL', "OpenGL", "绿通道 Y 朝上 (Blender/Unity 约定)"),
               ('DIRECTX', "DirectX", "绿通道 Y 朝下 (Unreal 约定)")),
        default='AUTO')
    smooth_px: FloatProperty(
        name="平滑 (px)", default=0.0, min=0.0, soft_max=16.0,
        description="高斯低通 σ(像素), 抑制噪点/锯齿(0=关)")
    deadzone_lsb: FloatProperty(
        name="平整死区 (LSB)", default=1.0, min=0.0, soft_max=4.0,
        description="法线 XY 分量低于该值(8bit 台阶数)视为纯平——"
                    "量化噪声经积分会放大成低频起伏, 死区保证平坦区严格为平(0=关)")
    slope_limit: FloatProperty(
        name="坡度上限", default=10.0, min=0.0, soft_max=20.0,
        description="梯度幅值限幅, 压制烘焙噪声/压缩伪影导致的尖刺(0=关)")
    backface_fix: BoolProperty(
        name="双面卡片背面反向", default=True,
        description="自动识别双面几何的背面(UV 绕向为负且与正面共面反法线), "
                    "位移取反使正背面同向跟随形状——发卡/布片类资产必开")
    reconstruct_z: BoolProperty(
        name="重建 Z 通道", default=False,
        description="双通道法线贴图(BC5 / AG 打包)从 XY 重建 Z")
    subdiv_mode: EnumProperty(
        name="细分方式",
        items=(('SIMPLE', "Simple (保形)", "保持低模形状, 细节与烘焙面完全对位(推荐)"),
               ('CATMULL_CLARK', "Catmull-Clark", "平滑基面(基面会轻微收缩, 与烘焙面略有偏差)"),
               ('LINEAR', "Linear", "线性细分")),
        default='SIMPLE')
    auto_levels: BoolProperty(
        name="自动级别", default=True,
        description="按贴图分辨率 × UV 占用率自动匹配 ≈1 四边形/texel")
    levels: IntProperty(
        name="级别", default=6, min=1, max=9,
        description="手动指定 Multires 细分级别")
    quad_budget: IntProperty(
        name="四边形预算", default=16_000_000, min=10_000, max=120_000_000,
        description="细分产生的四边形上限(内存保护); 自动与手动级别都受此约束")
    detrend: EnumProperty(
        name="UV 岛去趋势",
        items=(('PLANE', "平面 (推荐)",
                "逐岛去除最小二乘平面: 消除岛间台阶与岛内斜坡, 图集类贴图(发卡/部件拼图)必开"),
               ('MEAN', "均值",
                "逐岛去除高度均值: 只消岛间台阶; 独占 UV 的雕刻烘焙贴图可用"),
               ('OFF', "关", "不做逐岛修正")),
        default='PLANE')
    v_mode: EnumProperty(
        name="V 方向",
        items=(('AUTO', "自动", "按 UV 落点命中有效像素的比例自动判定(游戏 D3D 资产常需翻转)"),
               ('NORMAL', "正常", "贴图内容与 Blender UV 同向"),
               ('FLIP', "翻转", "贴图内容相对网格 UV 上下颠倒")),
        default='AUTO')
    highpass_px: FloatProperty(
        name="高通截止 (px)", default=0.0, min=0.0, soft_max=1024.0,
        description="抑制波长大于该像素数的大形态成分(0=关)。"
                    "图集贴图积分残留跨岛低频鼓包时可设 128~512")


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

        layout.template_ID(s, "image", open="image.open")
        layout.prop(s, "strength", slider=True)
        row = layout.row()
        row.scale_y = 1.2
        row.operator("nmtm.build", icon='MOD_MULTIRES')

        box = layout.box()
        box.label(text="选项", icon='PREFERENCES')
        box.prop(s, "subdiv_mode", text="细分")
        row = box.row()
        row.prop(s, "auto_levels")
        sub = row.row()
        sub.active = not s.auto_levels
        sub.prop(s, "levels", text="")
        box.prop(s, "quad_budget")
        box.prop(s, "backface_fix")
        box.prop(s, "detrend", text="去趋势")
        box.prop(s, "bump_mode", text="凹凸")
        box.prop(s, "v_mode", text="V 方向")
        box.prop(s, "green_mode", text="绿通道")
        box.prop(s, "highpass_px")
        box.prop(s, "smooth_px")
        box.prop(s, "deadzone_lsb")
        box.prop(s, "slope_limit")
        box.prop(s, "reconstruct_z")

        if obj is not None and obj.get("nmtm_owned"):
            box = layout.box()
            box.label(text="当前状态", icon='CHECKMARK')
            box.label(text=f"贴图: {obj.get('nmtm_image', '?')}")
            box.label(text=f"级别 {obj.get('nmtm_level', '?')} | "
                           f"强度 {obj.get('nmtm_strength', 0.0):.2f}")
            box.operator("nmtm.remove", icon='TRASH')


CLASSES = (NMTMSettings, NMTM_PT_panel)


def register_props():
    bpy.types.Scene.nmtm = PointerProperty(type=NMTMSettings)


def unregister_props():
    del bpy.types.Scene.nmtm
