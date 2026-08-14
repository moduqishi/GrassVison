"""本地确定性像素工具（不依赖视觉模型，数值精确）。

供服务端无感工具（grassvision_pixel_*）执行：
- dominant_colors: 区域主色调（精确 #RRGGBB + 份额）+ 候选色匹配
- pixel_diff:      同图两区域像素差异（百分比 + 最差子区域坐标）
- trace:           区域前景几何信息（包围盒 / 占比 / 边缘轨迹 SVG）
"""
from __future__ import annotations

import io

from PIL import Image, ImageChops, ImageFilter, ImageOps

MAX_SAMPLE = 96  # 分析前降采样的最大边长（够算色与差异，够快）


def _load(raw_bytes: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception:
        return None


def _region_box(img: Image.Image, region: str | None) -> tuple[int, int, int, int] | None:
    """把 0-1000 归一化区域 "x1,y1,x2,y2" 转成图像像素盒。"""
    if not region:
        return None
    try:
        parts = [int(p.strip()) for p in region.split(",")]
    except (ValueError, AttributeError):
        return None
    if len(parts) != 4:
        return None
    w, h = img.size
    x1 = max(0, int(parts[0] * w / 1000))
    y1 = max(0, int(parts[1] * h / 1000))
    x2 = min(w, int(parts[2] * w / 1000))
    y2 = min(h, int(parts[3] * h / 1000))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)


def _crop(img: Image.Image, box: tuple[int, int, int, int] | None) -> Image.Image:
    return img.crop(box) if box else img


def _hex(c) -> str:
    return "#%02X%02X%02X" % (c[0], c[1], c[2])


def _downsample(img: Image.Image, max_edge: int = MAX_SAMPLE) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    ratio = max_edge / max(w, h)
    return img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)


def dominant_colors(raw_bytes: bytes, region: str | None = None,
                    candidates: list[str] | None = None, top: int = 6) -> list[dict]:
    """返回区域主色调列表：[{color:'#RRGGBB', share:0.42}]，按份额降序。

    用 getcolors 统计精确像素色值（不量化，保留原始 #RRGGBB）。
    """
    img = _load(raw_bytes)
    if img is None:
        return []
    crop = _crop(img, _region_box(img, region))
    sample = _downsample(crop)
    counts: dict[tuple, int] = {}
    try:
        raw_counts = sample.getcolors(maxcolors=10 ** 6)
    except Exception:
        raw_counts = []
    if raw_counts:
        for count, color in raw_counts:
            if isinstance(color, int):  # 灰度图
                color = (color, color, color)
            counts[color] = counts.get(color, 0) + count
    else:
        for c in sample.getdata():
            counts[c] = counts.get(c, 0) + 1
    total = max(1, sum(counts.values()))
    colors = sorted(
        ((_hex(k), v / total) for k, v in counts.items()),
        key=lambda x: x[1], reverse=True,
    )[:top]
    result = [{"color": c, "share": round(s, 3)} for c, s in colors]
    if candidates:
        matched = _match_candidates(img, _region_box(img, region), candidates)
        result += [{"color": c, "share": 0.0, "matched": True} for c in matched]
    return result


def _match_candidates(img: Image.Image, box, candidates: list[str]) -> list[str]:
    """从候选色值里选出与区域像素最接近的（按平均色距排序）。"""
    crop = _crop(img, box)
    sample = _downsample(crop, 32)
    px = list(sample.getdata())
    if not px:
        return []
    try:
        cands = [
            (c.upper(), (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)))
            for c in candidates if isinstance(c, str) and len(c) == 7 and c.startswith("#")
        ]
    except (ValueError, IndexError):
        return []
    scored = []
    for label, rgb in cands:
        dist = sum(
            (p[0] - rgb[0]) ** 2 + (p[1] - rgb[1]) ** 2 + (p[2] - rgb[2]) ** 2
            for p in px[:: max(1, len(px) // 200)]
        )
        scored.append((dist, label))
    scored.sort()
    return [label for _, label in scored]


def pixel_diff_images(raw_a: bytes, raw_b: bytes,
                      region_a: str | None = None, region_b: str | None = None) -> dict:
    """对比两张图（或同图两区域）的像素差异。

    返回 {diff_percent, mean_diff, worst_box(0-1000归一化,基于A图), sizes}。
    区域尺寸不同时先缩放到同一尺寸再对比。
    """
    img_a = _load(raw_a)
    img_b = _load(raw_b)
    if img_a is None or img_b is None:
        return {"error": "图片解析失败"}
    box_a = _region_box(img_a, region_a)
    box_b = _region_box(img_b, region_b)
    if region_a and box_a is None:
        return {"error": "区域 A 无效，需 x1,y1,x2,y2（0-1000 归一化）"}
    if region_b and box_b is None:
        return {"error": "区域 B 无效，需 x1,y1,x2,y2（0-1000 归一化）"}
    crop_a = _crop(img_a, box_a)
    crop_b = _crop(img_b, box_b)
    a = _downsample(crop_a, 128)
    b = _downsample(crop_b, 128).resize(a.size, Image.LANCZOS)
    diff = ImageChops.difference(a, b)
    hist = diff.histogram()
    total_px = a.width * a.height
    # 差异像素 = 任一通道差 > 阈值
    threshold = 24
    n_diff = sum(hist[i] for i in range(threshold, 256))
    diff_percent = round(n_diff / max(1, total_px) * 100, 2)
    mean_diff = round(sum(i * hist[i] for i in range(256)) / max(1, total_px * 3), 2)

    # 分 4x4 块找最差子区域（基于 A 图 0-1000 归一化坐标）
    worst = {"share": -1, "box": None}
    bw, bh = max(1, a.width // 4), max(1, a.height // 4)
    base_box = box_a or (0, 0, img_a.width, img_a.height)
    for gy in range(4):
        for gx in range(4):
            tile = diff.crop((gx * bw, gy * bh, min((gx + 1) * bw, a.width),
                              min((gy + 1) * bh, a.height)))
            th = tile.histogram()
            share = sum(th[threshold:]) / max(1, tile.width * tile.height)
            if share > worst["share"]:
                worst["share"] = round(share * 100, 1)
                x1 = base_box[0] + (base_box[2] - base_box[0]) * (gx * bw) / a.width
                y1 = base_box[1] + (base_box[3] - base_box[1]) * (gy * bh) / a.height
                x2 = base_box[0] + (base_box[2] - base_box[0]) * min((gx + 1) * bw, a.width) / a.width
                y2 = base_box[1] + (base_box[3] - base_box[1]) * min((gy + 1) * bh, a.height) / a.height
                worst["box"] = [int(x1 * 1000 / img_a.width), int(y1 * 1000 / img_a.height),
                                int(x2 * 1000 / img_a.width), int(y2 * 1000 / img_a.height)]
    return {
        "diff_percent": diff_percent,
        "mean_diff": mean_diff,
        "worst_region": worst["box"] if worst["share"] >= 0 else None,
        "worst_share": worst["share"],
        "size": [a.width, a.height],
    }


def pixel_diff(raw_bytes: bytes, region_a: str, region_b: str) -> dict:
    """（兼容旧接口）对比同一图片两个区域的像素差异。"""
    return pixel_diff_images(raw_bytes, raw_bytes, region_a, region_b)


# ───────────────────────── 图元识别（确定性几何拟合）─────────────────────────
# 从二值 mask 提取可编辑 SVG 图元（circle/rect/line/polygon/path），
# 不依赖 LLM：连通分量 → 形状分类（圆度/实心度/细长度）→ 几何拟合。

import math as _math


def _mask_array(mask: Image.Image) -> bytearray:
    """mask → 一维 bytearray（0/255），逐像素访问比 getpixel 快。"""
    return bytearray(mask.tobytes())


def _connected_components(arr: bytearray, w: int, h: int, min_size: int = 6) -> list[set]:
    """BFS 洪水填充：把 mask 分成连通分量（过滤小噪点）。"""
    visited = set()
    comps = []
    for y in range(h):
        base = y * w
        for x in range(w):
            if arr[base + x] == 0 or (x, y) in visited:
                continue
            stack = [(x, y)]
            comp: set = set()
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in visited:
                    continue
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                if arr[cy * w + cx] == 0:
                    continue
                visited.add((cx, cy))
                comp.add((cx, cy))
                stack.append((cx + 1, cy))
                stack.append((cx - 1, cy))
                stack.append((cx, cy + 1))
                stack.append((cx, cy - 1))
            if len(comp) >= min_size:
                comps.append(comp)
    return comps


def _component_stats(comp: set) -> tuple:
    """返回 (bbox, area, 边缘像素集)。边缘 = 4-邻域含背景的像素。"""
    xs = [p[0] for p in comp]
    ys = [p[1] for p in comp]
    bbox = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
    area = len(comp)
    # 边缘像素：至少一个 4-邻域邻居不在 comp
    edges = {p for p in comp
             if (p[0] + 1, p[1]) not in comp or (p[0] - 1, p[1]) not in comp
             or (p[0], p[1] + 1) not in comp or (p[0], p[1] - 1) not in comp}
    return bbox, area, edges


def _fit_circle(pts: list) -> tuple | None:
    """Kasa 最小二乘圆拟合 → (cx, cy, r)。先中心化再解 2x2 正规方程。"""
    n = len(pts)
    if n < 5:
        return None
    mx = sum(x for x, y in pts) / n
    my = sum(y for x, y in pts) / n
    # 中心化坐标
    xs = [x - mx for x, y in pts]
    ys = [y - my for x, y in pts]
    rs2 = [x * x + y * y for x, y in zip(xs, ys)]
    sxx = sum(x * x for x in xs)
    syy = sum(y * y for y in ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxr = sum(x * r for x, r in zip(xs, rs2))
    syr = sum(y * r for y, r in zip(ys, rs2))
    a11, a12, a21, a22 = 2 * sxx, 2 * sxy, 2 * sxy, 2 * syy
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-9:
        return None
    dcx = (sxr * a22 - a12 * syr) / det
    dcy = (a11 * syr - sxr * a21) / det
    cx, cy = mx + dcx, my + dcy
    r = sum(_math.hypot(x - cx, y - cy) for x, y in pts) / n
    if r < 0.5:
        return None
    return cx, cy, r


def _convex_hull(points: list) -> list:
    """Andrew 单调链凸包 → 逆时针有序点。"""
    pts = sorted(set(points))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _douglas_peucker(points: list, epsilon: float) -> list:
    """Douglas-Peucker 折线简化。"""
    if len(points) <= 2:
        return points

    def _dist(p, a, b):
        x1, y1 = a
        x2, y2 = b
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return _math.hypot(p[0] - x1, p[1] - y1)
        t = max(0.0, min(1.0, ((p[0] - x1) * dx + (p[1] - y1) * dy) / (dx * dx + dy * dy)))
        return _math.hypot(p[0] - (x1 + t * dx), p[1] - (y1 + t * dy))

    dmax = 0.0
    idx = 0
    for i in range(1, len(points) - 1):
        d = _dist(points[i], points[0], points[-1])
        if d > dmax:
            idx, dmax = i, d
    if dmax > epsilon:
        left = _douglas_peucker(points[:idx + 1], epsilon)
        right = _douglas_peucker(points[idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def _skeleton_zhang_suen(arr: bytearray, w: int, h: int) -> set:
    """Zhang-Suen 细化 → 骨架像素集（返回新 mask 数组同构的集合）。"""
    # 工作副本（0/1）
    g = bytearray(1 if arr[i] > 0 else 0 for i in range(len(arr)))
    changed = True
    while changed:
        changed = False
        for step in (1, 2):
            to_remove = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    i = y * w + x
                    if g[i] != 1:
                        continue
                    p2, p3, p4 = g[i - w], g[i - w + 1], g[i + 1]
                    p5, p6, p7 = g[i + w + 1], g[i + w], g[i + w - 1]
                    p8, p9 = g[i - 1], g[i - w - 1]
                    neighbors = (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)
                    if not (2 <= neighbors <= 6):
                        continue
                    # 0→1 穿越次数（8 邻域循环）
                    seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                    trans = sum(1 for k in range(8) if seq[k] == 0 and seq[k + 1] == 1)
                    if trans != 1:
                        continue
                    if step == 1 and (p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0):
                        to_remove.append(i)
                    if step == 2 and (p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0):
                        to_remove.append(i)
            for i in to_remove:
                if g[i] == 1:
                    g[i] = 0
                    changed = True
    return {(x, y) for y in range(h) for x in range(w) if g[y * w + x] == 1}


def _order_skeleton_path(skeleton: set) -> list:
    """把骨架点排成有序路径（从端点沿邻居走；分叉取最长支）。"""
    if not skeleton:
        return []
    # 度 = 8 邻域骨架邻居数
    def _neighbors(p):
        x, y = p
        return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx or dy) and (x + dx, y + dy) in skeleton]

    ends = [p for p in skeleton if len(_neighbors(p)) <= 1]
    if not ends:
        ends = [next(iter(skeleton))]
    best_path = []
    for start in ends:
        path = [start]
        visited = {start}
        while True:
            nxt = [n for n in _neighbors(path[-1]) if n not in visited]
            if not nxt:
                break
            path.append(nxt[0])
            visited.add(nxt[0])
        if len(path) > len(best_path):
            best_path = path
    return best_path


def _classify_and_fit(comp: set, bbox: tuple) -> dict | None:
    """组件 → {type, ...} 图元（mask 采样坐标）。"""
    bx1, by1, bx2, by2 = bbox
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 3 or bh < 3:
        return None
    _, area, edges = _component_stats(comp)
    perimeter = len(edges)
    solidity = area / (bw * bh)
    roundness = (4 * _math.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
    aspect = max(bw, bh) / max(1, min(bw, bh))

    # 1) 矩形优先：非细长 + 高实心度（细长矩形=粗线/进度条，交给骨架线分支）
    if aspect <= 2.5 and solidity > 0.82:
        return {"type": "rect", "x": bx1, "y": by1, "width": bw, "height": bh}

    # 2) 圆检测：圆度 + 实心（实心圆 solidity≈0.78，roundness>0.9）
    if roundness > 0.8 and solidity > 0.55:
        fit = _fit_circle(sorted(edges))
        if fit:
            cx, cy, r = fit
            if 0.35 <= r / max(bw, bh) <= 1.6:
                return {"type": "circle", "cx": round(cx, 2), "cy": round(cy, 2),
                        "r": round(r, 2)}

    # 3) 细长组件：骨架 → 线/折线（由 recognize_shapes 骨架分支处理）
    if aspect > 2.5:
        return None

    # 4) 多边形：凸包 + DP 简化
    hull = _convex_hull(list(comp))
    if len(hull) >= 3:
        eps = max(1.0, min(bw, bh) * 0.08)
        poly = _douglas_peucker(hull, eps)
        if len(poly) >= 3:
            return {"type": "polygon", "points": poly}
    return {"type": "path", "points": list(comp)[:64]}


def recognize_shapes(mask: Image.Image) -> list[dict]:
    """二值 mask → 图元列表（mask 采样坐标，SVG 片段 + 描述）。

    返回 [{type, description, svg(片段)}]，无图元时返回 []。
    """
    w, h = mask.size
    arr = _mask_array(mask)
    comps = _connected_components(arr, w, h)
    shapes: list[dict] = []
    # 先做组件级分类
    for comp in comps:
        bbox, area, _edges = _component_stats(comp)
        item = _classify_and_fit(comp, bbox)
        if item is None:
            continue
        shapes.append((comp, item))
    # 细长组件：骨架 → 线/折线（单独处理，避免长骨架被漏掉）
    handled = {id(c) for c, _ in shapes}
    for comp in comps:
        if id(comp) in handled:
            continue
        bx1, by1, bx2, by2 = _component_stats(comp)[0]
        bw, bh = bx2 - bx1, by2 - by1
        if max(bw, bh) / max(1, min(bw, bh)) <= 2.5:
            continue  # 不是细长，且未被分类（可能太碎）→ 跳过
        # 该组件单独成 mask → 骨架
        w2, h2 = bx2 - bx1 + 2, by2 - by1 + 2
        sub = bytearray(w2 * h2)
        for (x, y) in comp:
            sub[(y - by1 + 1) * w2 + (x - bx1 + 1)] = 255
        skeleton = _skeleton_zhang_suen(sub, w2, h2)
        path = _order_skeleton_path(skeleton)
        if len(path) < 3:
            continue
        # 映射回全局坐标
        path = [(p[0] + bx1 - 1, p[1] + by1 - 1) for p in path]
        # 直线检测：端点连线最大偏离
        p0, pn = path[0], path[-1]
        dev = max(_point_line_dist(p, p0, pn) for p in path) if len(path) > 2 else 0.0
        if dev <= max(1.5, min(bw, bh) * 0.15):
            shapes.append((comp, {"type": "line", "x1": p0[0], "y1": p0[1],
                                  "x2": pn[0], "y2": pn[1]}))
        else:
            # 折线：DP 简化（epsilon 取大些，滤掉骨架噪点分支）
            eps = max(2.0, min(bw, bh) * 0.2)
            poly = _douglas_peucker(path, eps)
            if len(poly) >= 2:
                shapes.append((comp, {"type": "polyline", "points": poly}))
    return [s for _, s in shapes]


def _point_line_dist(p, a, b):
    x1, y1 = a
    x2, y2 = b
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return _math.hypot(p[0] - x1, p[1] - y1)
    t = max(0.0, min(1.0, ((p[0] - x1) * dx + (p[1] - y1) * dy) / (dx * dx + dy * dy)))
    return _math.hypot(p[0] - (x1 + t * dx), p[1] - (y1 + t * dy))


def _shape_to_svg(item: dict, w: int, h: int) -> str:
    """图元 dict → SVG 片段（w/h 为画布尺寸）。"""
    t = item["type"]
    if t == "circle":
        return f'<circle cx="{item["cx"]}" cy="{item["cy"]}" r="{item["r"]}"/>'
    if t == "rect":
        return f'<rect x="{item["x"]}" y="{item["y"]}" width="{item["width"]}" height="{item["height"]}"/>'
    if t == "line":
        return f'<line x1="{item["x1"]}" y1="{item["y1"]}" x2="{item["x2"]}" y2="{item["y2"]}" stroke="#000" stroke-width="2"/>'
    if t in ("polygon", "polyline"):
        pts = " ".join(f"{int(p[0])},{int(p[1])}" for p in item["points"])
        tag = "polygon" if t == "polygon" else "polyline"
        return f'<{tag} points="{pts}" fill="none" stroke="#000" stroke-width="2"/>'
    if t == "path":
        pts = " ".join(f"{int(p[0])},{int(p[1])}" for p in item["points"])
        return f'<polyline points="{pts}" fill="none" stroke="#000" stroke-width="2"/>'
    return ""


def _shape_desc(item: dict) -> str:
    t = item["type"]
    if t == "circle":
        return f"圆形：圆心({item['cx']},{item['cy']}) 半径{item['r']}"
    if t == "rect":
        return f"矩形：位置({item['x']},{item['y']}) 尺寸{item['width']}×{item['height']}"
    if t == "line":
        return f"线段：({item['x1']},{item['y1']})→({item['x2']},{item['y2']})"
    if t in ("polygon", "polyline"):
        n = len(item["points"])
        return f"{'多边形' if t == 'polygon' else '折线'}：{n} 个顶点 {item['points'][:4]}…"
    return "复杂路径（边缘轨迹）"


def trace_region(raw_bytes: bytes, region: str | None = None, max_points: int = 200) -> dict:
    """提取区域前景几何信息（不依赖视觉模型）。

    返回 {foreground_box(原图像素), width, height, coverage(前景占比),
          dominant_color, svg(可编辑图元：circle/rect/line/polygon，复杂形状回退轨迹),
          shapes(图元列表), description(几何摘要文本)}。
    仅适用于扁平高对比图形；照片等复杂内容会给出整体占比。
    """
    img = _load(raw_bytes)
    if img is None:
        return {"error": "图片解析失败"}
    box = _region_box(img, region)
    crop = _crop(img, box)
    orig_scale = (img.width, img.height)
    sample = _downsample(crop, 160)
    gray = sample.convert("L")
    # 前景 = 与区域主色距离较远的像素（简单二值化：亮度偏离中值）
    px = list(gray.getdata())
    mid = sorted(px)[len(px) // 2]
    threshold = 30
    mask = gray.point(lambda v: 255 if abs(v - mid) > threshold else 0)
    fg = mask.getbbox()
    coverage = 0.0
    fg_box_px = None
    if fg:
        coverage = round(
            sum(1 for v in mask.crop(fg).getdata() if v > 0)
            / max(1, mask.crop(fg).width * mask.crop(fg).height), 3
        )
        scale_x = (box[2] - box[0]) / sample.width if box else img.width / sample.width
        scale_y = (box[3] - box[1]) / sample.height if box else img.height / sample.height
        ox = box[0] if box else 0
        oy = box[1] if box else 0
        fg_box_px = [int(ox + fg[0] * scale_x), int(oy + fg[1] * scale_y),
                     int(ox + fg[2] * scale_x), int(oy + fg[3] * scale_y)]
    # 主色：用前景包围盒（转 0-1000 归一化）
    dom = []
    if fg_box_px:
        fg_region = ",".join(str(int(v * 1000 / s)) for v, s in zip(
            fg_box_px, [img.width, img.height, img.width, img.height]))
        dom = dominant_colors(raw_bytes, fg_region, top=1)
    # 图元识别：连通分量 → 形状分类 → 几何拟合 → 可编辑 SVG
    shapes_raw = recognize_shapes(mask)
    # 坐标映射：mask 采样坐标 → 原图像素
    def _to_orig(x, y):
        sx = (box[2] - box[0]) / sample.width if box else img.width / sample.width
        sy = (box[3] - box[1]) / sample.height if box else img.height / sample.height
        ox = box[0] if box else 0
        oy = box[1] if box else 0
        return round(ox + x * sx), round(oy + y * sy)

    shapes = []
    for item in shapes_raw:
        it = dict(item)
        if it["type"] == "circle":
            cx, cy = _to_orig(it["cx"], it["cy"])
            sx = (box[2] - box[0]) / sample.width if box else img.width / sample.width
            it["cx"], it["cy"], it["r"] = cx, cy, round(it["r"] * sx, 2)
        elif it["type"] == "rect":
            x1, y1 = _to_orig(it["x"], it["y"])
            x2, y2 = _to_orig(it["x"] + it["width"], it["y"] + it["height"])
            it["x"], it["y"], it["width"], it["height"] = x1, y1, x2 - x1, y2 - y1
        elif it["type"] == "line":
            x1, y1 = _to_orig(it["x1"], it["y1"])
            x2, y2 = _to_orig(it["x2"], it["y2"])
            it["x1"], it["y1"], it["x2"], it["y2"] = x1, y1, x2, y2
        elif it["type"] in ("polygon", "polyline", "path"):
            it["points"] = [_to_orig(px, py) for px, py in it["points"]]
        it["description"] = _shape_desc(it)
        it["svg"] = _shape_to_svg(it, img.width, img.height)
        shapes.append(it)

    if shapes:
        svg_parts = "".join(s["svg"] for s in shapes)
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">%s</svg>'
            % (img.width, img.height, svg_parts)
        )
        descriptions = "；".join(s["description"] for s in shapes)
    else:
        # 兜底：边缘轨迹 polyline（原图坐标）
        edges = mask.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > 60 else 0)
        pts = []
        ew, eh = edges.size
        step = max(1, (ew * eh) // (max_points * 4))
        idx = 0
        for y in range(eh):
            for x in range(ew):
                idx += 1
                if idx % step != 0:
                    continue
                if edges.getpixel((x, y)) > 0:
                    pts.append(_to_orig(x, y))
                    if len(pts) >= max_points:
                        break
            if len(pts) >= max_points:
                break
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">'
            '<polyline fill="none" stroke="#000" stroke-width="1" points="%s"/></svg>'
            % (img.width, img.height, " ".join(f"{p[0]},{p[1]}" for p in pts))
        ) if pts else ""
        descriptions = "复杂图形（边缘轨迹，未能拟合成基本图元）"
    return {
        "foreground_box_px": fg_box_px,
        "width": fg_box_px[2] - fg_box_px[0] if fg_box_px else 0,
        "height": fg_box_px[3] - fg_box_px[1] if fg_box_px else 0,
        "coverage": coverage,
        "dominant_color": dom[0]["color"] if dom else None,
        "shapes": shapes,
        "svg": svg,
        "description": descriptions,
        "original_size": list(orig_scale),
    }

def render_html(html: str, width: int = 1280, height: int = 800, wait_ms: int = 300) -> bytes:
    """用无头 Chrome/Chromium 渲染 HTML，返回 PNG 字节。

    可执行文件探测：环境变量 GRASSVISION_CHROME 优先，其次常见系统路径。
    容器部署需安装 chromium（见 Dockerfile）。
    """
    import os
    import subprocess
    import tempfile

    env_chrome = os.environ.get("GRASSVISION_CHROME", "")
    candidates = [
        env_chrome,
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
    ]
    exe = next((c for c in candidates if c and os.path.exists(c)), None)
    if not exe:
        raise RuntimeError("未找到 Chrome/Chromium（可设置 GRASSVISION_CHROME 环境变量）")

    with tempfile.TemporaryDirectory() as d:
        html_path = os.path.join(d, "page.html")
        out_path = os.path.join(d, "shot.png")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html or "<html><body></body></html>")
        cmd = [
            exe, "--headless", "--disable-gpu", "--hide-scrollbars",
            f"--window-size={max(64, width)},{max(64, height)}",
            f"--screenshot={out_path}",
            f"--virtual-time-budget={max(50, wait_ms)}",
            f"file://{html_path}",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=40)
        if result.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"Chrome 渲染失败: {result.stderr.decode(errors='replace')[:200]}")
        with open(out_path, "rb") as f:
            return f.read()
