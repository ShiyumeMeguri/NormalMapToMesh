# -*- coding: utf-8 -*-
# NormalMapToMesh 数学内核 —— 纯 numpy, 不依赖 bpy, 可用 `python core.py` 独立自测。
#
# v2 烘焙位移管线: 切线基交给渲染器——在物体的真实材质上用 Cycles 烘焙
# "物体空间法线"(Bake Type=Normal, Space=Object), 每个像素的 mikktspace 切线、
# 绿通道约定、通道重建网络(游戏材质常见)全部由渲染器按材质节点求值,
# 天然正确处理数百个任意旋转、镜像、混合手性的 UV 岛。
# 本内核只做纯数学部分: 双线性采样 + Displace(RGB_TO_XYZ) 同款向量位移 + 逐顶点平均。
#
# v1 的频域泊松积分已废弃: 它把法线贴图当单一高度场在 UV 平面上全局积分,
# 隐含"UV 轴与切线基全局对齐"假设——平面/单岛测试成立, 真实游戏资产
# (实测发型: 406 个 UV 岛 / 144 个镜像岛 / 逐岛任意旋转的切线帧)上
# 梯度方向逐岛错乱, 属于前提性失败, 不可修补。

import numpy as np


def sample_bilinear_wrap(field, u, v):
    """按 UV 双线性采样; field 形状 (H, W) 或 (H, W, C); u/v 任意实数, 平铺 wrap。"""
    h, w = field.shape[:2]
    x = u * np.float32(w) - np.float32(0.5)
    y = v * np.float32(h) - np.float32(0.5)
    x0f = np.floor(x)
    y0f = np.floor(y)
    tx = (x - x0f).astype(np.float32)
    ty = (y - y0f).astype(np.float32)
    x0 = x0f.astype(np.int64) % w
    y0 = y0f.astype(np.int64) % h
    x1 = x0 + 1
    x1[x1 == w] = 0
    y1 = y0 + 1
    y1[y1 == h] = 0
    if field.ndim == 3:
        tx = tx[:, None]
        ty = ty[:, None]
    a = field[y0, x0]
    b = field[y0, x1]
    c = field[y1, x0]
    d = field[y1, x1]
    return (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty


def displace_rgb_to_xyz(rgb_loop, strength, mid_level=0.5):
    """Blender Displace 修改器 RGB_TO_XYZ 方向的同款位移: (rgb − mid) × strength。

    rgb_loop (N, 3) 为烘焙的物体空间法线像素([0,1] 编码), 输出 (N, 3) 物体空间
    位移向量。物体空间法线单位长, 位移幅值恒为 strength/2, 起伏来自逐像素方向差
    ——这正是"法线转位移"目视等效的核心; 双面卡片正背面共享 texel 时得到同一
    向量、同向移动, 薄片不会被撕开。
    """
    return ((rgb_loop - np.float32(mid_level)) * np.float32(strength)).astype(np.float32)


def average_loop_vectors_to_verts(loop_vecs, loop_vert, vert_count):
    """逐 loop 向量 → 逐顶点平均(UV 接缝顶点自动取两侧均值)。(N,3) → (V,3)。"""
    cnt = np.maximum(np.bincount(loop_vert, minlength=vert_count), 1)
    out = np.empty((vert_count, 3), np.float32)
    for c in range(3):
        s = np.bincount(loop_vert, weights=loop_vecs[:, c].astype(np.float64),
                        minlength=vert_count)
        out[:, c] = (s / cnt).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 自测: python core.py
# ---------------------------------------------------------------------------

def _selftest():
    rng = np.random.default_rng(7)
    h = w = 64
    field = rng.uniform(0.0, 1.0, (h, w, 3)).astype(np.float32)

    # 像素中心处应精确还原(多通道路径)
    iu = ((np.arange(w) + 0.5) / w).astype(np.float32)
    iv = np.full(w, (17 + 0.5) / h, np.float32)
    got = sample_bilinear_wrap(field, iu, iv)
    err = np.abs(got - field[17]).max()
    print(f"[双线性] 像素中心最大误差 = {err:.2e}")
    assert err < 1e-5

    # wrap 寻址: u 与 u+1 等价
    got2 = sample_bilinear_wrap(field, iu + 1.0, iv)
    assert np.abs(got2 - got).max() < 1e-5

    # 单通道 (H, W) 路径兼容
    got1 = sample_bilinear_wrap(field[..., 0], iu, iv)
    assert np.abs(got1 - field[17, :, 0]).max() < 1e-5

    # Displace RGB_TO_XYZ: 中性像素(0.5)零位移; (1,0.5,0.5) → (+s/2,0,0); 绿0 → (0,-s/2,0)
    rgb = np.array([[0.5, 0.5, 0.5], [1.0, 0.5, 0.5], [0.5, 0.0, 0.5]], np.float32)
    off = displace_rgb_to_xyz(rgb, 0.02)
    assert np.abs(off[0]).max() < 1e-7
    assert np.allclose(off[1], [0.01, 0.0, 0.0], atol=1e-7)
    assert np.allclose(off[2], [0.0, -0.01, 0.0], atol=1e-7)
    print(f"[位移] RGB_TO_XYZ 校验通过: {off[1]} / {off[2]}")

    # 逐顶点平均: 顶点1被两个 loop 引用 → 取均值
    lv = np.array([0, 1, 1], np.int32)
    vecs = np.array([[1, 0, 0], [0, 2, 0], [0, 4, 2]], np.float32)
    avg = average_loop_vectors_to_verts(vecs, lv, 2)
    assert np.allclose(avg[0], [1, 0, 0]) and np.allclose(avg[1], [0, 3, 1])
    print(f"[顶点平均] 接缝均值校验通过: {avg[1]}")

    print("core.py 自测全部通过 ✓")


if __name__ == "__main__":
    _selftest()
