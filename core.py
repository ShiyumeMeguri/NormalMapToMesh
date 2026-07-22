# -*- coding: utf-8 -*-
# NormalMapToMesh 数学内核 —— 纯 numpy, 不依赖 bpy, 可用 `python core.py` 独立自测。
#
# v3 烘焙高度重建管线: 切线基交给渲染器——用 Cycles EMIT 烘焙三张物体空间图:
#   n1 = 带法线贴图的扰动法线(材质 Normal Map 节点输出, 世界→物体空间编码)
#   n0 = 纯平滑基准法线(Geometry.Normal)
#   P  = 表面位置(Geometry.Position, 物体空间, 原始浮点)
# 每 texel 的高度梯度零猜测装配: 3D 坡度 g = n0 − n1/(n1·n0) (切平面向量, |g|=tanθ),
#   dh/du = g·∂P/∂u, dh/dv = g·∂P/∂v (∂P 由 P 图中心差分得到)。
# 镜像 UV 岛/逐岛任意旋转切线帧/绿通道约定/通道重建网络全部由烘焙数据自动携带,
# 且梯度直接是"物体单位/UV", 积分出的高度就是物理高度(倍数 1.0 = 真实浮雕)。
# 之后走 Frankot-Chellappa 频域泊松积分 → 逐岛去趋势 → 岛间缝合 → 沿 n0 位移。
#
# 历史教训:
# - v1 直接按"UV 轴=切线轴"解读贴图梯度做全局积分——真实资产逐岛切线帧任意旋转
#   +镜像混合手性, 前提性失败。
# - v2 差分位移(n1−n0)×scale——幅值与细节波长无关, 高频发丝沟壑位移超过顶点间距,
#   表面撕碎; 坡度必须先积分成高度(高频自动得到小高度)才物理正确。

import numpy as np

# ---------------------------------------------------------------------------
# 采样与解码
# ---------------------------------------------------------------------------


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


def decode_unit_normal(rgb):
    """[0,1] 编码法线 → 单位向量 + 有效权重。

    权重 = 解码后长度接近 1 才为 1(烘焙背景黑 → (-1,-1,-1) 长度 1.73 → 0),
    防未烘焙 texel 的垃圾值污染。返回 (n 单位化, w)。
    """
    n = rgb.astype(np.float32) * 2.0 - 1.0
    ln = np.sqrt(np.einsum('...i,...i->...', n, n))
    w = (np.abs(ln - 1.0) < 0.35).astype(np.float32)
    n /= np.maximum(ln, 1e-6)[..., None]
    return n, w


# ---------------------------------------------------------------------------
# 高度梯度装配(零约定猜测)
# ---------------------------------------------------------------------------

def height_gradients(rgb_detail, rgb_base, pos, min_cos=0.2):
    """三张烘焙图 → UV 域高度梯度 (gx=dh/du, gy=dh/dv, 物体单位) + 有效权重。

    g = n0 − n1/(n1·n0): 高度场表面梯度的 3D 形式(切平面向量, |g| = tanθ);
    ∂P/∂u、∂P/∂v 用位置图中心差分——镜像岛的 U 轴自动反向, 混合手性零处理。
    dh/du = g·∂P/∂u 直接携带每 texel 的真实世界尺度(逐岛密度差异自动正确)。
    岛间沟槽处两侧 margin 相遇会产生巨大 |∂P| 假差分, 用稳健中位数阈值剔除。
    """
    n1, w1 = decode_unit_normal(rgb_detail)
    n0, w0 = decode_unit_normal(rgb_base)
    dot = np.einsum('...i,...i->...', n1, n0)
    w = w1 * w0 * (dot > min_cos).astype(np.float32)
    g = n0 - n1 / np.maximum(dot, min_cos)[..., None]

    hgt, wid = dot.shape
    pu = (np.roll(pos, -1, axis=1) - np.roll(pos, 1, axis=1)) * (wid / 2.0)
    pv = (np.roll(pos, -1, axis=0) - np.roll(pos, 1, axis=0)) * (hgt / 2.0)
    # 中心差分要求两侧邻 texel 也有效
    wu = w * np.roll(w, -1, axis=1) * np.roll(w, 1, axis=1)
    wv = w * np.roll(w, -1, axis=0) * np.roll(w, 1, axis=0)

    lu = np.sqrt(np.einsum('...i,...i->...', pu, pu))
    lv = np.sqrt(np.einsum('...i,...i->...', pv, pv))
    vu = lu[wu > 0.0]
    vv = lv[wv > 0.0]
    if vu.size:
        wu = wu * (lu < 16.0 * max(float(np.median(vu)), 1e-12)).astype(np.float32)
    if vv.size:
        wv = wv * (lv < 16.0 * max(float(np.median(vv)), 1e-12)).astype(np.float32)

    gx = (np.einsum('...i,...i->...', g, pu) * wu).astype(np.float32)
    gy = (np.einsum('...i,...i->...', g, pv) * wv).astype(np.float32)
    return gx, gy, w


# ---------------------------------------------------------------------------
# Frankot-Chellappa 频域泊松积分
# ---------------------------------------------------------------------------

def integrate_height(gx, gy, highpass_wavelength=0.0, smooth_sigma=0.0):
    """最小二乘可积化: 给定梯度场求高度场 (全局零均值→平地锚定)。

    梯度为 dh/du(物体单位/UV), 积分域取 UV 单位正方形(像素间距 1/W, 1/H),
    返回高度即物体单位的物理高度。O(N logN), 4K 贴图约 1~2s。
    highpass_wavelength: >0 时抑制波长(UV单位)大于该值的成分(跨岛低频鼓包)。
    smooth_sigma: >0 时做高斯低通(σ, UV单位), 抑制采样锯齿/噪点。
    """
    h, w = gx.shape
    wx = (2.0 * np.pi) * np.fft.rfftfreq(w, d=1.0 / w)   # rad / UV单位
    wy = (2.0 * np.pi) * np.fft.fftfreq(h, d=1.0 / h)
    gx_f = np.fft.rfft2(gx)
    gy_f = np.fft.rfft2(gy)
    denom = wx[None, :] ** 2 + wy[:, None] ** 2
    denom[0, 0] = 1.0
    hf = (wx[None, :] * gx_f + wy[:, None] * gy_f) * (-1j)
    hf /= denom
    hf[0, 0] = 0.0
    if highpass_wavelength > 0.0:
        kc = (2.0 * np.pi) / highpass_wavelength
        k2 = (wx[None, :] ** 2 + wy[:, None] ** 2) / (kc * kc)
        hf *= 1.0 - np.exp(-(k2 * k2))   # 4阶高斯高通, 拐点较陡
    if smooth_sigma > 0.0:
        k_sq = wx[None, :] ** 2 + wy[:, None] ** 2
        hf *= np.exp(-0.5 * smooth_sigma * smooth_sigma * k_sq)
    out = np.fft.irfft2(hf, s=(h, w)).astype(np.float32)
    # 基准面锚定: 平坦区(零梯度, 含未烘焙背景)应为 0 高度
    flat = (gx == 0.0) & (gy == 0.0)
    if flat.mean() > 0.01:
        out -= np.float32(out[flat].mean())
    else:
        out -= np.float32(np.median(out))
    return out


# ---------------------------------------------------------------------------
# UV 岛: 面级并查集 + 逐岛去趋势 + 岛间缝合 (v1 机器, 实测自洽)
# ---------------------------------------------------------------------------

def face_islands(loop_vert, loop_uv, poly_of_loop, poly_count):
    """按 (顶点, 量化UV) 归并面 → UV 岛标签。

    共享同一顶点且 UV 重合的两个 loop 所属的面判为同岛(覆盖共边与顶点粘连)。
    返回 (labels[poly_count] int32, 岛数)。
    """
    qu = np.round(loop_uv[:, 0] * 65536.0).astype(np.int64)
    qv = np.round(loop_uv[:, 1] * 65536.0).astype(np.int64)
    lv = loop_vert.astype(np.int64)
    order = np.lexsort((qv, qu, lv))
    lv_s, qu_s, qv_s = lv[order], qu[order], qv[order]
    pp = poly_of_loop[order]
    same = (lv_s[1:] == lv_s[:-1]) & (qu_s[1:] == qu_s[:-1]) & (qv_s[1:] == qv_s[:-1])

    parent = np.arange(poly_count, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]  # 路径减半
            a = parent[a]
        return a

    for i in np.nonzero(same)[0]:
        ra, rb = find(pp[i]), find(pp[i + 1])
        if ra != rb:
            parent[rb] = ra

    roots = np.array([find(p) for p in range(poly_count)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32), int(labels.max()) + 1 if poly_count else 0


def detrend_per_island(h_loop, uv_loop, island_of_loop, n_islands, mode='PLANE'):
    """逐 UV 岛去趋势: 消除岛间高度台阶与岛内积分残留的斜坡。

    全局 FFT 积分会让每个岛带上任意常量偏移(MEAN 消除)和跨岛泄漏的线性斜坡
    (PLANE 消除)。island_of_loop 来自基面拓扑映射, 精确且不怕 UV 重叠。
    PLANE 用逐岛中心化最小二乘平面拟合, 全向量化。
    """
    if n_islands < 1 or mode == 'OFF':
        return h_loop
    isl = island_of_loop
    cnt = np.bincount(isl, minlength=n_islands).astype(np.float64)
    cnt_safe = np.maximum(cnt, 1.0)
    h64 = h_loop.astype(np.float64)
    mean_h = np.bincount(isl, weights=h64, minlength=n_islands) / cnt_safe
    if mode == 'MEAN':
        med = _segment_median(h64, isl, n_islands, cnt)
        return (h64 - med[isl]).astype(np.float32)

    # PLANE: h ≈ a·u + b·v + c, 逐岛中心化后 2x2 正规方程
    u = uv_loop[:, 0].astype(np.float64)
    v = uv_loop[:, 1].astype(np.float64)
    mean_u = np.bincount(isl, weights=u, minlength=n_islands) / cnt_safe
    mean_v = np.bincount(isl, weights=v, minlength=n_islands) / cnt_safe
    du = u - mean_u[isl]
    dv = v - mean_v[isl]
    dh = h64 - mean_h[isl]
    suu = np.bincount(isl, weights=du * du, minlength=n_islands)
    svv = np.bincount(isl, weights=dv * dv, minlength=n_islands)
    suv = np.bincount(isl, weights=du * dv, minlength=n_islands)
    suh = np.bincount(isl, weights=du * dh, minlength=n_islands)
    svh = np.bincount(isl, weights=dv * dh, minlength=n_islands)
    # 岭系数防细长条岛退化(u,v 近共线)
    ridge = 1e-10 * np.maximum(suu + svv, 1e-30)
    det = (suu + ridge) * (svv + ridge) - suv * suv
    det = np.where(np.abs(det) < 1e-30, 1.0, det)
    a = (suh * (svv + ridge) - svh * suv) / det
    b = (svh * (suu + ridge) - suh * suv) / det
    resid = dh - a[isl] * du - b[isl] * dv
    # 斜坡去除后再用中位数锚定常量(凸起细节不拉偏基准面)
    med = _segment_median(resid, isl, n_islands, cnt)
    return (resid - med[isl]).astype(np.float32)


def _segment_median(values, isl, n_islands, cnt):
    """逐岛中位数(向量化): 按 (岛, 值) 排序后取每段中点。"""
    order = np.lexsort((values, isl))
    starts = np.zeros(n_islands, np.int64)
    np.cumsum(cnt.astype(np.int64), out=starts)
    starts -= cnt.astype(np.int64)          # 每岛段起点
    mid = starts + (cnt.astype(np.int64) - 1) // 2
    mid = np.clip(mid, 0, max(len(values) - 1, 0))
    med = values[order[mid]] if len(values) else np.zeros(n_islands)
    med = np.where(cnt > 0, med, 0.0)
    return med


def stitch_islands(h_loop, loop_vert, island_of_loop, n_islands):
    """岛间缝合: 最小二乘求每岛常量修正, 使相邻岛在共享网格顶点处高度对齐。

    以"同一网格顶点上不同岛的高度均值应相等"为约束建 n_islands 维正规方程,
    岭正则固定全局自由度。全向量化, 岛数级别的稠密解, 开销可忽略。
    """
    if n_islands < 2:
        return h_loop
    ni = np.int64(n_islands)
    key = loop_vert.astype(np.int64) * ni + island_of_loop
    uniq, inv = np.unique(key, return_inverse=True)
    sums = np.bincount(inv, weights=h_loop.astype(np.float64))
    cnts = np.bincount(inv)
    group_mean = sums / cnts
    gv = uniq // ni          # 组所属顶点
    gi = (uniq % ni).astype(np.int64)   # 组所属岛
    # uniq 按 key 有序 → 同顶点的组相邻; 相邻对即缝合约束
    same_vert = gv[1:] == gv[:-1]
    a = gi[:-1][same_vert]
    b = gi[1:][same_vert]
    r = (group_mean[:-1] - group_mean[1:])[same_vert]   # h̄_a - h̄_b
    if a.shape[0] == 0:
        return h_loop
    deg = (np.bincount(a, minlength=n_islands)
           + np.bincount(b, minlength=n_islands)).astype(np.float64)
    rhs = np.zeros(n_islands, np.float64)
    np.add.at(rhs, a, -r)
    np.add.at(rhs, b, r)
    if n_islands <= 4096:
        mat = np.zeros((n_islands, n_islands), np.float64)
        np.add.at(mat, (a, a), 1.0)
        np.add.at(mat, (b, b), 1.0)
        np.add.at(mat, (a, b), -1.0)
        np.add.at(mat, (b, a), -1.0)
        mat[np.diag_indices(n_islands)] += 1e-6 + 1e-9 * deg.max()
        c = np.linalg.solve(mat, rhs)
    else:
        # 岛数过大时用 Jacobi 迭代(拉普拉斯系统, 收敛快且全向量化)
        c = np.zeros(n_islands, np.float64)
        dd = deg + 1e-6
        for _ in range(128):
            s = (np.bincount(a, weights=c[b], minlength=n_islands)
                 + np.bincount(b, weights=c[a], minlength=n_islands))
            c = (rhs + s) / dd
    c -= c.mean()
    return (h_loop.astype(np.float64) + c[island_of_loop]).astype(np.float32)


# ---------------------------------------------------------------------------
# 逐顶点归并
# ---------------------------------------------------------------------------

def average_loops_to_verts(loop_vals, loop_vert, vert_count):
    """逐 loop 标量 → 逐顶点平均(接缝顶点自动取两侧均值)。"""
    s = np.bincount(loop_vert, weights=loop_vals.astype(np.float64), minlength=vert_count)
    c = np.bincount(loop_vert, minlength=vert_count)
    return (s / np.maximum(c, 1)).astype(np.float32)


def average_loop_vectors_to_verts(loop_vecs, loop_vert, vert_count):
    """逐 loop 向量 → 逐顶点平均。(N,3) → (V,3)。"""
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

    # ---- 双线性采样 ----
    h = w = 64
    field = rng.uniform(0.0, 1.0, (h, w, 3)).astype(np.float32)
    iu = ((np.arange(w) + 0.5) / w).astype(np.float32)
    iv = np.full(w, (17 + 0.5) / h, np.float32)
    got = sample_bilinear_wrap(field, iu, iv)
    err = np.abs(got - field[17]).max()
    print(f"[双线性] 像素中心最大误差 = {err:.2e}")
    assert err < 1e-5
    got2 = sample_bilinear_wrap(field, iu + 1.0, iv)
    assert np.abs(got2 - got).max() < 1e-5
    got1 = sample_bilinear_wrap(field[..., 0], iu, iv)
    assert np.abs(got1 - field[17, :, 0]).max() < 1e-5

    # ---- 端到端: 平面片高度场 → (n1, n0, P) 烘焙图合成 → 梯度装配 → 积分还原 ----
    # 平面绕 Z 转 30°(模拟"UV 轴 ≠ 物体轴"), 尺度 A×B 各不相同(模拟逐岛密度)
    w2 = h2 = 256
    A, B = 2.0, 1.3
    ang = np.deg2rad(30.0)
    ex = np.array([np.cos(ang), np.sin(ang), 0.0])
    ey = np.array([-np.sin(ang), np.cos(ang), 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    uu = (np.arange(w2, dtype=np.float64) + 0.5) / w2
    vv = (np.arange(h2, dtype=np.float64) + 0.5) / h2
    ug, vg = np.meshgrid(uu, vv)
    height = np.zeros((h2, w2))
    for _ in range(5):
        ku, kv = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        amp = float(rng.uniform(0.003, 0.01))
        ph = float(rng.uniform(0, 2 * np.pi))
        height += amp * np.sin(2 * np.pi * (ku * ug + kv * vg) + ph)
    # 边界渐落窗: 真实烘焙的图像边缘是无内容背景; 环绕差分在边界列会被
    # 稳健阈值剔除(梯度归零), 内容必须在边界处平坦才与算法前提一致
    ramp = np.minimum(np.minimum(ug, 1.0 - ug), np.minimum(vg, 1.0 - vg)) / 0.1
    window = 0.5 - 0.5 * np.cos(np.pi * np.clip(ramp, 0.0, 1.0))
    height = (height - height.mean()) * window
    dh_du = np.gradient(height, axis=1) * w2
    dh_dv = np.gradient(height, axis=0) * h2

    pos = (ug[..., None] * (A * ex) + vg[..., None] * (B * ey)).astype(np.float32)
    n0_vec = np.broadcast_to(ez, (h2, w2, 3))
    # 表面梯度(3D): dh/dx·ex + dh/dy·ey, 其中 dh/dx = dh/du / A
    grad3d = (dh_du / A)[..., None] * ex + (dh_dv / B)[..., None] * ey
    n1_vec = n0_vec - grad3d
    n1_vec = n1_vec / np.linalg.norm(n1_vec, axis=-1, keepdims=True)
    enc = lambda n: (n.astype(np.float32) + 1.0) * 0.5
    gx, gy, wmask = height_gradients(enc(n1_vec), enc(np.array(n0_vec)), pos)
    assert wmask.mean() > 0.99, "合成图应全部有效"
    rec = integrate_height(gx, gy).astype(np.float64)
    diff = rec - height
    diff -= diff.mean()
    rel = np.sqrt(np.mean(diff ** 2)) / np.sqrt(np.mean(height ** 2))
    print(f"[端到端] 旋转+异尺度平面 相对RMS误差 = {rel:.2e}  (阈值 2e-2)")
    assert rel < 2e-2, "梯度装配/积分还原失败"

    # 镜像岛(U 轴反向)——P 图携带反向, 结果仍应正确
    pos_m = ((1.0 - ug)[..., None] * (A * ex) + vg[..., None] * (B * ey)).astype(np.float32)
    grad3d_m = (-dh_du / A)[..., None] * ex + (dh_dv / B)[..., None] * ey
    n1_m = n0_vec - grad3d_m
    n1_m = n1_m / np.linalg.norm(n1_m, axis=-1, keepdims=True)
    gxm, gym, _ = height_gradients(enc(n1_m), enc(np.array(n0_vec)), pos_m)
    rec_m = integrate_height(gxm, gym).astype(np.float64)
    diff_m = rec_m - height
    diff_m -= diff_m.mean()
    rel_m = np.sqrt(np.mean(diff_m ** 2)) / np.sqrt(np.mean(height ** 2))
    print(f"[镜像岛] 相对RMS误差 = {rel_m:.2e}")
    assert rel_m < 2e-2, "镜像 UV 岛未被 P 图自动纠正"

    # 未烘焙背景(黑)与岛间大跳变: 权重应归零
    n1_bad = enc(n1_vec).copy()
    n1_bad[:8] = 0.0
    gxb, gyb, wb = height_gradients(n1_bad, enc(np.array(n0_vec)), pos)
    assert wb[:8].max() == 0.0 and np.abs(gxb[:4]).max() == 0.0, "背景未归零"

    # ---- FC 积分基准面锚定 ----
    r2 = (ug - 0.5) ** 2 + (vg - 0.5) ** 2
    bump = 0.05 * np.exp(-r2 / 0.005)
    gx_b = (np.gradient(bump, axis=1) * w2).astype(np.float32)
    gy_b = (np.gradient(bump, axis=0) * h2).astype(np.float32)
    gx_b[np.abs(gx_b) < 1e-6] = 0.0
    gy_b[np.abs(gy_b) < 1e-6] = 0.0
    hb = integrate_height(gx_b, gy_b).astype(np.float64)
    corner_lvl = hb[:32, :32].mean()
    peak = hb.max()
    print(f"[锚定] 平地电平 = {corner_lvl:+.2e}  峰值 = {peak:.4f} (真值 0.05)")
    assert abs(corner_lvl) < 1e-3 and abs(peak - 0.05) < 0.005

    # ---- UV 岛并查集 ----
    loop_uv = np.array([
        [0.05, 0.05], [0.45, 0.05], [0.45, 0.45], [0.05, 0.45],
        [0.55, 0.55], [0.95, 0.55], [0.95, 0.95], [0.55, 0.95],
    ], np.float32)
    loop_vert = np.array([0, 1, 2, 3, 4, 5, 6, 7], np.int32)
    poly_of_loop = np.array([0, 0, 0, 0, 1, 1, 1, 1], np.int64)
    labels, n_isl = face_islands(loop_vert, loop_uv, poly_of_loop, 2)
    assert n_isl == 2 and labels[0] != labels[1]
    lv2 = np.array([0, 1, 2, 2, 1, 3], np.int32)
    uv2 = np.array([[0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 1]], np.float32)
    pol2 = np.array([0, 0, 0, 1, 1, 1], np.int64)
    _, n2 = face_islands(lv2, uv2, pol2, 2)
    assert n2 == 1
    print("[UV岛] 并查集校验通过")

    # ---- 逐岛去趋势 ----
    m = 4000
    rng2 = np.random.default_rng(3)
    u0 = rng2.uniform(0, 0.4, m)
    v0 = rng2.uniform(0, 0.4, m)
    sine = 0.002 * np.sin(2 * np.pi * 40 * u0)
    h0 = 3.0 * u0 - 2.0 * v0 + 0.7 + sine
    u1 = rng2.uniform(0.6, 0.9, m)
    v1 = rng2.uniform(0.6, 0.9, m)
    h1 = np.full(m, -5.0)
    hh_loop = np.concatenate([h0, h1]).astype(np.float32)
    uv_loop2 = np.stack([np.concatenate([u0, u1]),
                         np.concatenate([v0, v1])], axis=1).astype(np.float32)
    isl_loop = np.concatenate([np.zeros(m, np.int64), np.ones(m, np.int64)])
    out = detrend_per_island(hh_loop, uv_loop2, isl_loop, 2, 'PLANE')
    resid0 = out[:m] - (sine - sine.mean())
    assert np.abs(resid0).max() < 5e-4 and np.abs(out[m:]).max() < 1e-6
    print(f"[去趋势] 岛0残差 = {np.abs(resid0).max():.2e}")

    # ---- 岛间缝合 ----
    lv3 = np.array([0, 1, 1, 2, 2, 3], np.int64)
    il3 = np.array([0, 0, 1, 1, 2, 2], np.int64)
    hh3 = np.array([1.0, 1.0, 4.0, 4.0, -2.0, -2.0], np.float32)
    out3 = stitch_islands(hh3, lv3, il3, 3)
    assert abs(out3[2] - out3[1]) < 1e-3 and abs(out3[4] - out3[3]) < 1e-3
    print("[缝合] 岛间台阶对齐通过")

    # ---- 顶点归并 ----
    lv = np.array([0, 1, 1], np.int32)
    avg = average_loops_to_verts(np.array([1.0, 2.0, 4.0], np.float32), lv, 2)
    assert np.allclose(avg, [1.0, 3.0])
    vecs = np.array([[1, 0, 0], [0, 2, 0], [0, 4, 2]], np.float32)
    avg3 = average_loop_vectors_to_verts(vecs, lv, 2)
    assert np.allclose(avg3[1], [0, 3, 1])
    print("[顶点平均] 标量/向量归并通过")

    print("core.py 自测全部通过 ✓")


if __name__ == "__main__":
    _selftest()
