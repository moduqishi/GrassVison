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


def pixel_diff(raw_bytes: bytes, region_a: str, region_b: str) -> dict:
    """对比同一图片两个区域的像素差异。

    返回 {diff_percent, mean_diff, worst_box(0-1000归一化), sizes}。
    区域尺寸不同时先缩放到同一尺寸再对比。
    """
    img = _load(raw_bytes)
    if img is None:
        return {"error": "图片解析失败"}
    box_a = _region_box(img, region_a)
    box_b = _region_box(img, region_b)
    if not box_a or not box_b:
        return {"error": "区域无效，需 x1,y1,x2,y2（0-1000 归一化）"}
    a = _downsample(_crop(img, box_a), 128)
    b = _downsample(_crop(img, box_b), 128).resize(a.size, Image.LANCZOS)
    diff = ImageChops.difference(a, b)
    hist = diff.histogram()
    total_px = a.width * a.height
    # 差异像素 = 任一通道差 > 阈值
    threshold = 24
    n_diff = sum(
        hist[i] for i in range(threshold, 256)
    )
    diff_percent = round(n_diff / max(1, total_px) * 100, 2)
    mean_diff = round(sum(i * hist[i] for i in range(256)) / max(1, total_px * 3), 2)

    # 分 4x4 块找最差子区域（返回 0-1000 归一化坐标）
    worst = {"share": -1, "box": None}
    bw, bh = max(1, a.width // 4), max(1, a.height // 4)
    for gy in range(4):
        for gx in range(4):
            tile = diff.crop((gx * bw, gy * bh, min((gx + 1) * bw, a.width),
                              min((gy + 1) * bh, a.height)))
            th = tile.histogram()
            share = sum(th[threshold:]) / max(1, tile.width * tile.height)
            if share > worst["share"]:
                worst["share"] = round(share * 100, 1)
                # 映射回原图 0-1000 坐标
                x1 = box_a[0] + (box_a[2] - box_a[0]) * (gx * bw) / a.width
                y1 = box_a[1] + (box_a[3] - box_a[1]) * (gy * bh) / a.height
                x2 = box_a[0] + (box_a[2] - box_a[0]) * (min((gx + 1) * bw, a.width)) / a.width
                y2 = box_a[1] + (box_a[3] - box_a[1]) * (min((gy + 1) * bh, a.height)) / a.height
                worst["box"] = [int(x1 * 1000 / img.width), int(y1 * 1000 / img.height),
                                int(x2 * 1000 / img.width), int(y2 * 1000 / img.height)]
    return {
        "diff_percent": diff_percent,
        "mean_diff": mean_diff,
        "worst_region": worst["box"] if worst["share"] >= 0 else None,
        "worst_share": worst["share"],
        "size": [a.width, a.height],
    }


def trace_region(raw_bytes: bytes, region: str | None = None, max_points: int = 200) -> dict:
    """提取区域前景几何信息（不依赖视觉模型）。

    返回 {foreground_box(原图像素), width, height, coverage(前景占比),
          dominant_color, svg(polyline 边缘轨迹，原图坐标)}。
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
    # 边缘轨迹：提取边缘像素，按行采样为 polyline（原图坐标）
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
                scale_x = (box[2] - box[0]) / ew if box else img.width / ew
                scale_y = (box[3] - box[1]) / eh if box else img.height / eh
                ox = box[0] if box else 0
                oy = box[1] if box else 0
                pts.append((round(ox + x * scale_x), round(oy + y * scale_y)))
                if len(pts) >= max_points:
                    break
        if len(pts) >= max_points:
            break
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">'
        '<polyline fill="none" stroke="#000" stroke-width="1" points="%s"/></svg>'
        % (img.width, img.height, " ".join(f"{p[0]},{p[1]}" for p in pts))
    ) if pts else ""
    return {
        "foreground_box_px": fg_box_px,
        "width": fg_box_px[2] - fg_box_px[0] if fg_box_px else 0,
        "height": fg_box_px[3] - fg_box_px[1] if fg_box_px else 0,
        "coverage": coverage,
        "dominant_color": dom[0]["color"] if dom else None,
        "svg": svg,
        "original_size": list(orig_scale),
    }
