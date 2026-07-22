# -*- coding: utf-8 -*-
bl_info = {
    "name": "Normal Map To Mesh (法线→多级精度高模)",
    "author": "Ruri",
    "version": (2, 0, 0),
    "blender": (4, 2, 0),
    "location": "3D视图 > 侧栏(N) > Ruri",
    "description": "把法线贴图经 Cycles 物体空间法线烘焙 + Displace(RGB→XYZ) 同款向量位移 "
                   "重建为 Multires 高模细节, 一键自动细分+位移+写回; 切线基/绿通道约定/"
                   "镜像 UV 岛全部由渲染器按材质真实节点求值",
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
