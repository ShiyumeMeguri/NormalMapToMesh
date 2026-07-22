# -*- coding: utf-8 -*-
# NormalMapToMesh 数学内核 —— 纯 numpy, 不依赖 bpy, 可用 `python core.py` 独立自测。
#
# 管线: 切线空间法线贴图 → 梯度场 → Frankot-Chellappa 频域泊松积分 → 高度场
#       → 按细分网格逐 loop 双线性采样 → UV 岛去均值(消岛间台阶) → 逐顶点平均 → 沿法线位移。
# 全程向量化, 无逐像素/逐顶点 Python 循环(UV岛光栅化按三角形分块向量化)。

import numpy as np

# ---------------------------------------------------------------------------
# 法线解码
# ---------------------------------------------------------------------------

def valid_mask(pixels, width, height, channels):
    """有效法线像素掩码: alpha>0.5 且解码后长度正常且 nz>0。(H, W) bool。

    alpha 通道整体为空(全≈0)时视为不透明——部分导出管线/新建图不写 alpha。
    """
    img = pixels.reshape(height, width, channels)
    n = img[..., :3].astype(np.float32) * 2.0 - 1.0
    len_sq = np.einsum('ijk,ijk->ij', n, n)
    valid = (len_sq > 0.25) & (len_sq < 2.25) & (n[..., 2] > 0.05)
    if channels >= 4:
        a = img[..., 3]
        if float(a.max()) > 0.5:
            valid &= a > 0.5
    return valid


def decode_normals(pixels, width, height, channels,
                   flip_green=False, reconstruct_z=False, v_flip=False,
                   deadzone=0.0):
    """把平铺 float32 像素解码成切线空间梯度场 (gx, gy)。

    gx = dh/dx, gy = dh/dy —— 高度对"世界切线单位"的斜率(调用方再乘 texel 世界尺度)。
    无效像素(alpha为0 / 长度异常 / nz<=0)梯度置 0, 泊松积分会把这些区域视为平地。
    v_flip: 贴图内容相对网格 UV 上下颠倒(D3D 约定资产)时翻转行序;
            翻转行序会反转 y 轴, 因此 ny 同步取反, 保证梯度在网格 UV 系里自洽。
    deadzone: |nx|/|ny| 低于该值(归一化 [-1,1] 单位)时视为纯平——8bit 量化的
              ±1 LSB 噪声经泊松积分会放大成低频起伏, 死区让平坦区严格为平。
    返回 (gx, gy) 均为 float32 (H, W), 行序与网格 V 轴对齐。
    """
    img = pixels.reshape(height, width, channels)
    if v_flip:
        img = img[::-1]
    n = img[..., :3].astype(np.float32, copy=True)
    n *= 2.0
    n -= 1.0
    if v_flip:
        n[..., 1] = -n[..., 1]
    if flip_green:
        n[..., 1] = -n[..., 1]
    if reconstruct_z:
        # 双通道(BC5/AG)贴图: 从 xy 重建 z
        n[..., 2] = np.sqrt(np.maximum(0.0, 1.0 - n[..., 0] ** 2 - n[..., 1] ** 2))
    if deadzone > 0.0:
        n[..., 0][np.abs(n[..., 0]) <= deadzone] = 0.0
        n[..., 1][np.abs(n[..., 1]) <= deadzone] = 0.0

    len_sq = np.einsum('ijk,ijk->ij', n, n)
    valid = (len_sq > 0.25) & (len_sq < 2.25) & (n[..., 2] > 0.05)
    if channels >= 4:
        a = img[..., 3]
        if float(a.max()) > 0.5:   # alpha 全空视为不透明
            valid &= a > 0.5

    inv_nz = np.where(valid, 1.0 / np.maximum(n[..., 2], 0.05), 0.0).astype(np.float32)
    gx = -n[..., 0] * inv_nz
    gy = -n[..., 1] * inv_nz
    return gx, gy


# ---------------------------------------------------------------------------
# Frankot-Chellappa 频域泊松积分
# ---------------------------------------------------------------------------

def height_skewness(field, valid, stride=4):
    """有效区高度分布偏度。细节凸起为主 → 右偏(>0); 左偏说明高度符号反了。

    不同生态的法线贴图存在整体凹凸约定差异(等价于高度取负),
    可积性无法区分全局符号, 用"细节以凸起为主"的统计先验判定。
    """
    f = field[::stride, ::stride].astype(np.float64)
    m = valid[::stride, ::stride]
    v = f[m]
    if v.size < 16:
        return 0.0
    v = v - v.mean()
    s2 = np.mean(v * v)
    if s2 <= 0.0:
        return 0.0
    return float(np.mean(v ** 3) / s2 ** 1.5)


def clamp_slope(gx, gy, limit):
    """坡度限幅: |g| 超过 limit 的像素等比压回。压制烘焙噪声/压缩伪影的尖刺。"""
    if limit <= 0.0:
        return gx, gy
    mag = np.hypot(gx, gy)
    scale = np.minimum(1.0, limit / np.maximum(mag, 1e-12)).astype(np.float32)
    return gx * scale, gy * scale


def curl_residual(gx, gy, stride=4):
    """梯度场旋度残差 ∂gy/∂x − ∂gx/∂y 的均方值。

    真实高度场的梯度无旋; 绿通道符号错误会使 gy 取反、旋度显著增大,
    以此自动判定 OpenGL/DirectX 约定。降采样评估, 开销可忽略。
    """
    a = gx[::stride, ::stride].astype(np.float64)
    b = gy[::stride, ::stride].astype(np.float64)
    d = np.gradient(b, axis=1) - np.gradient(a, axis=0)
    return float(np.mean(d * d))


def integrate_height(gx, gy, highpass_wavelength=0.0, smooth_sigma=0.0):
    """最小二乘可积化: 给定梯度场求高度场 (全局零均值)。

    梯度以"世界切线单位"计, 积分域取 UV 单位正方形(像素间距 1/W, 1/H),
    因此返回高度的量纲 = 世界高度 / texel世界尺度 S, 调用方乘 S 还原真实高度。
    highpass_wavelength: >0 时抑制波长(UV单位)大于该值的成分——图集类贴图
    (发卡/部件拼图)全局积分会产生跨岛低频鼓包, 用高通只保留细节。
    smooth_sigma: >0 时做高斯低通(σ, UV单位), 抑制采样锯齿/噪点。
    O(N logN), 4K 贴图约 1~2s。
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
    # 基准面锚定: 平坦区(零梯度)应为 0 高度——全图零均值会被凸起把平地压负。
    flat = (gx == 0.0) & (gy == 0.0)
    if flat.mean() > 0.01:
        out -= np.float32(out[flat].mean())
    else:
        out -= np.float32(np.median(out))
    return out


# ---------------------------------------------------------------------------
# 双线性采样(wrap 寻址, 与周期性 FFT 一致)
# ---------------------------------------------------------------------------

def sample_bilinear_wrap(field, u, v):
    """按 UV 双线性采样高度场; u/v 任意实数, 平铺 wrap。返回 float32。"""
    h, w = field.shape
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
    a = field[y0, x0]
    b = field[y0, x1]
    c = field[y1, x0]
    d = field[y1, x1]
    return (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty


# ---------------------------------------------------------------------------
# UV 岛: 面级并查集 + 岛 id 光栅化 + 逐岛去均值
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

    图集类贴图(发卡/部件拼图)做全局泊松积分时, 每个岛会带上一个任意的
    常量偏移(MEAN 消除)和跨岛泄漏产生的线性斜坡(PLANE 消除)。
    island_of_loop 来自基面拓扑映射(细分面按基面连续分块), 精确且不怕 UV 重叠。
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
        # 常量用中位数锚定: 凸起细节不会把岛的基准面拉偏(均值会)
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
    """岛间缝合: 最小二乘求每岛常量修正, 使网格上相邻岛在共享顶点处高度对齐。

    去趋势会让每个岛獲得独立的基准面, 长发/部件被切成多段 UV 岛时段间出现台阶。
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


def average_loops_to_verts(loop_vals, loop_vert, vert_count):
    """逐 loop 值 → 逐顶点平均(接缝顶点自动取两侧均值)。"""
    s = np.bincount(loop_vert, weights=loop_vals.astype(np.float64), minlength=vert_count)
    c = np.bincount(loop_vert, minlength=vert_count)
    return (s / np.maximum(c, 1)).astype(np.float32)


# ---------------------------------------------------------------------------
# 自测: python core.py
# ---------------------------------------------------------------------------

def _selftest():
    rng = np.random.default_rng(7)
    w = h = 512
    scale = 2.5  # 假设的 texel 世界尺度 S

    # 合成高度场: 若干正弦叠加(解析梯度, 无离散化误差)
    u = (np.arange(w, dtype=np.float64) + 0.5) / w
    v = (np.arange(h, dtype=np.float64) + 0.5) / h
    uu, vv = np.meshgrid(u, v)
    height = np.zeros((h, w))
    dh_du = np.zeros((h, w))
    dh_dv = np.zeros((h, w))
    for _ in range(6):
        ku, kv = int(rng.integers(1, 12)), int(rng.integers(1, 12))
        amp = float(rng.uniform(0.005, 0.02))
        ph = float(rng.uniform(0, 2 * np.pi))
        arg = 2 * np.pi * (ku * uu + kv * vv) + ph
        height += amp * np.sin(arg)
        dh_du += amp * 2 * np.pi * ku * np.cos(arg)
        dh_dv += amp * 2 * np.pi * kv * np.cos(arg)
    height -= height.mean()

    # 世界斜率 → 切线空间法线 → 编码为贴图像素
    gx_world = dh_du / scale
    gy_world = dh_dv / scale
    nrm = np.stack([-gx_world, -gy_world, np.ones_like(gx_world)], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    px = np.empty((h, w, 4), np.float32)
    px[..., :3] = (nrm * 0.5 + 0.5).astype(np.float32)
    px[..., 3] = 1.0

    gx, gy = decode_normals(px.ravel(), w, h, 4)
    rec = integrate_height(gx, gy).astype(np.float64) * scale
    diff = rec - height
    diff -= diff.mean()          # 积分只定义到常数差
    ref = np.sqrt(np.mean(height ** 2))
    rel = np.sqrt(np.mean(diff ** 2)) / ref
    print(f"[FC积分] 相对RMS误差 = {rel:.2e}  (阈值 1e-2)")
    assert rel < 1e-2, "泊松积分误差超阈值"

    # 双线性采样: 像素中心处应精确还原
    field = rec.astype(np.float32)
    iu = ((np.arange(w) + 0.5) / w).astype(np.float32)
    got = sample_bilinear_wrap(field, iu, np.full(w, (17 + 0.5) / h, np.float32))
    err = np.abs(got - field[17]).max()
    print(f"[双线性] 像素中心最大误差 = {err:.2e}")
    assert err < 1e-4

    # wrap 寻址: u 与 u+1 等价
    got2 = sample_bilinear_wrap(field, iu + 1.0, np.full(w, (17 + 0.5) / h, np.float32))
    assert np.abs(got2 - got).max() < 1e-5

    # V 翻转往返: 模拟 D3D 约定资产(行序颠倒 + 绿通道表达在翻转帧里),
    # v_flip 解码后积分结果应与原始高度一致
    px_flip = px[::-1].copy()
    px_flip[..., 1] = 1.0 - px_flip[..., 1]
    gxf, gyf = decode_normals(px_flip.ravel(), w, h, 4, v_flip=True)
    rec_f = integrate_height(gxf, gyf).astype(np.float64) * scale
    diff_f = rec_f - height
    diff_f -= diff_f.mean()
    rms_f = np.sqrt(np.mean(diff_f ** 2)) / ref
    print(f"[V翻转] 相对RMS误差 = {rms_f:.2e}")
    assert rms_f < 1e-2, "v_flip 解码不自洽"

    # 基准面锚定: 中央凸起+大片平地 → 平地应精确归零(不被凸起压负)
    r2 = (uu - 0.5) ** 2 + (vv - 0.5) ** 2
    bump = 0.05 * np.exp(-r2 / 0.005)
    gx_b = (np.gradient(bump, axis=1) * w / scale).astype(np.float32)
    gy_b = (np.gradient(bump, axis=0) * h / scale).astype(np.float32)
    gx_b[np.abs(gx_b) < 1e-6] = 0.0
    gy_b[np.abs(gy_b) < 1e-6] = 0.0
    hb = integrate_height(gx_b, gy_b).astype(np.float64) * scale
    corner_lvl = hb[:64, :64].mean()
    peak = hb.max()
    print(f"[锚定] 平地电平 = {corner_lvl:+.2e}  峰值 = {peak:.4f} (真值 0.05)")
    assert abs(corner_lvl) < 1e-3, "平坦区未锚定到 0"
    assert abs(peak - 0.05) < 0.005, "凸起高度失真"

    # 高通: 低频(k=1)被压制, 高频(k=50)保留
    uu32, vv32 = np.meshgrid(u.astype(np.float64), v.astype(np.float64))
    h_low = 0.01 * np.sin(2 * np.pi * uu32)
    h_high = 0.01 * np.sin(2 * np.pi * 50 * uu32)
    for hh, keep_min, keep_max, tag in ((h_low, 0.0, 0.35, "低频"), (h_high, 0.9, 1.1, "高频")):
        g_u = np.gradient(hh, axis=1) * w      # dh/du
        gx_t = (g_u / scale).astype(np.float32)
        gy_t = np.zeros_like(gx_t)
        out = integrate_height(gx_t, gy_t, highpass_wavelength=0.25) * scale
        ratio_amp = out.std() / hh.std()
        print(f"[高通] {tag}保留率 = {ratio_amp:.2f}")
        assert keep_min <= ratio_amp <= keep_max, f"高通{tag}行为异常"

    # UV 岛并查集: 两个孤立面 → 2 岛; 共享顶点同 UV → 1 岛
    loop_uv = np.array([
        [0.05, 0.05], [0.45, 0.05], [0.45, 0.45], [0.05, 0.45],
        [0.55, 0.55], [0.95, 0.55], [0.95, 0.95], [0.55, 0.95],
    ], np.float32)
    loop_vert = np.array([0, 1, 2, 3, 4, 5, 6, 7], np.int32)
    poly_of_loop = np.array([0, 0, 0, 0, 1, 1, 1, 1], np.int64)
    labels, n_isl = face_islands(loop_vert, loop_uv, poly_of_loop, 2)
    assert n_isl == 2 and labels[0] != labels[1], "两个孤立面应为两岛"
    lv2 = np.array([0, 1, 2, 2, 1, 3], np.int32)
    uv2 = np.array([[0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 1]], np.float32)
    pol2 = np.array([0, 0, 0, 1, 1, 1], np.int64)
    _, n2 = face_islands(lv2, uv2, pol2, 2)
    assert n2 == 1, "共享UV边的两面应为一岛"

    # 逐岛去趋势: 岛0 = 平面+高频正弦, 岛1 = 常量偏移; PLANE 后只剩正弦
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
    print(f"[去趋势] 岛0残差 = {np.abs(resid0).max():.2e}  岛1残差 = {np.abs(out[m:]).max():.2e}")
    assert np.abs(resid0).max() < 5e-4, "PLANE 去趋势没消掉平面"
    assert np.abs(out[m:]).max() < 1e-6, "常量岛应归零"
    out_mean = detrend_per_island(hh_loop, uv_loop2, isl_loop, 2, 'MEAN')
    assert abs(np.median(out_mean[:m])) < 5e-3 and abs(np.median(out_mean[m:])) < 1e-6

    # 岛间缝合: 3 岛链, 岛1/岛2 带人为台阶, 共享顶点处应对齐
    #   顶点: 0..3; 岛0={loop于v0,v1}, 岛1={v1,v2}, 岛2={v2,v3}
    lv3 = np.array([0, 1, 1, 2, 2, 3], np.int64)
    il3 = np.array([0, 0, 1, 1, 2, 2], np.int64)
    hh3 = np.array([1.0, 1.0, 4.0, 4.0, -2.0, -2.0], np.float32)  # 台阶 +3 / -6
    out3 = stitch_islands(hh3, lv3, il3, 3)
    step01 = abs(out3[2] - out3[1])
    step12 = abs(out3[4] - out3[3])
    print(f"[缝合] 台阶残差 = {step01:.2e}, {step12:.2e}")
    assert step01 < 1e-3 and step12 < 1e-3, "岛间缝合未对齐台阶"

    # 旋度判据: 正确符号的梯度场旋度应远小于取反后的
    r_ok = curl_residual(gx, gy)
    r_bad = curl_residual(gx, -gy)
    print(f"[旋度] 正确 {r_ok:.2e} vs 反绿 {r_bad:.2e}")
    assert r_ok * 10 < r_bad, "旋度判据无法区分绿通道符号"

    print("core.py 自测全部通过 ✓")


if __name__ == "__main__":
    _selftest()
