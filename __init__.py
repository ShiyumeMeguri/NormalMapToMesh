# -*- coding: utf-8 -*-
bl_info = {
    "name": "Normal Map To Mesh (法线→多级精度高模)",
    "author": "Ruri",
    "version": (5, 3, 0),
    "blender": (4, 2, 0),
    "location": "3D视图 > 侧栏(N) > Ruri",
    "description": "法线贴图 → Multires 高模(免烘焙直算): mikktspace 切线帧光栅化 + "
                   "numpy 材质法线链求值 + 逐三角形解析 ∂P 装配高度梯度, 频域泊松积分出"
                   "物理高度, 逐岛去趋势+缝合+边缘锁定后沿真正 Catmull-Clark 细分曲面位移"
                   "(默认); 不支持的材质节点自动回退 Cycles EMIT 烘焙。平贴严格零位移, 倍数可调",
    "category": "Mesh",
}

import importlib

from . import core, ops, ui

# F8 / 重装 addon 时热重载
for _m in (core, ops, ui):
    importlib.reload(_m)

import bpy  # noqa: E402


def register():
    for cls in ui.CLASSES:
        bpy.utils.register_class(cls)
    for cls in ops.CLASSES:
        bpy.utils.register_class(cls)
    ui.register_props()


def unregister():
    ui.unregister_props()
    for cls in reversed(ops.CLASSES):
        bpy.utils.unregister_class(cls)
    for cls in reversed(ui.CLASSES):
        bpy.utils.unregister_class(cls)
    ops.clear_caches()
