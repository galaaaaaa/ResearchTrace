"""图/表/公式多模态阅读：区域检测（本地几何启发式）→ PNG 渲染 → VLM 结构化解读。

除 analyze_figure 内部的 VLM 调用外全部本地完成，不发网络请求。
所有入口不抛异常：失败降级返回空列表 / None 并写 tracer 事件。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from src.schemas import FigureReading
from src.settings import get_settings
from src.tracing import get_active_tracer
from src.utils import sha256_text


@dataclass
class FigureRegion:
    page: int
    bbox: tuple[float, float, float, float]
    kind: str          # figure / table / formula
    area_ratio: float


_CLUSTER_GAP_PT = 20.0        # 间距小于该值的元素 union 成一簇
_MIN_ELEMENT_PT = 3.0         # 更小的退化元素（细线、圆点等）直接丢弃
_TABLE_WORD_DENSITY = 0.0008  # 词数/pt²：高于该密度且数字词多 → 判为表格
_TABLE_NUMERIC_RATIO = 0.2    # 数字词占全部词的比例下限
_RENDER_DPI = 150


def _rects_near(a: pymupdf.Rect, b: pymupdf.Rect, gap: float) -> bool:
    """两矩形相交，或在任一方向上的间隙小于 gap（含对角接近）。"""
    ox = min(a.x1, b.x1) - max(a.x0, b.x0)  # >0 表示 x 方向有重叠
    oy = min(a.y1, b.y1) - max(a.y0, b.y0)
    if ox > 0 and oy > 0:
        return True
    if ox > 0:
        return -oy < gap
    if oy > 0:
        return -ox < gap
    return (-ox < gap) and (-oy < gap)


def _cluster_rects(rects: list[pymupdf.Rect], *, gap: float) -> list[pymupdf.Rect]:
    """贪心聚类：间距小于 gap 的矩形不断 union，直至稳定。"""
    clusters = [pymupdf.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        merged: list[pymupdf.Rect] = []
        for rect in clusters:
            target = None
            for i, existing in enumerate(merged):
                if _rects_near(rect, existing, gap):
                    target = i
                    break
            if target is None:
                merged.append(pymupdf.Rect(rect))
            else:
                merged[target] = merged[target] | rect
                changed = True
        clusters = merged
    return clusters


def _looks_like_table(page: pymupdf.Page, rect: pymupdf.Rect) -> bool:
    """bbox 内词密度高且数字词占比高 → 表格，否则按图形处理。"""
    try:
        words = page.get_text("words", clip=rect)
    except Exception:
        return False
    if not words:
        return False
    area = abs(rect.width * rect.height)
    if area <= 0:
        return False
    if len(words) / area < _TABLE_WORD_DENSITY:
        return False
    numeric = sum(1 for w in words if any(ch.isdigit() for ch in str(w[4])))
    return numeric / len(words) >= _TABLE_NUMERIC_RATIO


def detect_figure_regions(pdf_path: str, *, max_figures: int | None = None) -> list[FigureRegion]:
    """每页收集光栅图 bbox + 矢量矩形，聚类后按面积过滤，返回前 max_figures 个（面积降序）。"""
    tracer = get_active_tracer()
    try:
        s = get_settings()
        if max_figures is None:
            max_figures = int(s.source("vision", "max_figures_per_paper", 4) or 4)
        min_ratio = float(s.source("vision", "min_figure_area_ratio", 0.05) or 0.05)
        regions: list[FigureRegion] = []
        with pymupdf.open(pdf_path) as pdf:
            for pno in range(pdf.page_count):
                page = pdf[pno]
                page_area = abs(page.rect.width * page.rect.height) or 1.0
                boxes: list[pymupdf.Rect] = []
                for info in page.get_image_info():
                    boxes.append(pymupdf.Rect(info.get("bbox", (0, 0, 0, 0))))
                for drawing in page.get_drawings():
                    boxes.append(pymupdf.Rect(drawing.get("rect", (0, 0, 0, 0))))
                # 丢弃退化小元素，并裁剪到页面范围内
                clipped: list[pymupdf.Rect] = []
                for r in boxes:
                    if abs(r.width) < _MIN_ELEMENT_PT or abs(r.height) < _MIN_ELEMENT_PT:
                        continue
                    c = r & page.rect
                    if not c.is_empty and abs(c.width) >= _MIN_ELEMENT_PT and abs(c.height) >= _MIN_ELEMENT_PT:
                        clipped.append(c)
                for cluster in _cluster_rects(clipped, gap=_CLUSTER_GAP_PT):
                    ratio = abs(cluster.width * cluster.height) / page_area
                    if ratio < min_ratio:
                        continue  # 单个小元素 / 噪声被面积比过滤
                    kind = "table" if _looks_like_table(page, cluster) else "figure"
                    regions.append(
                        FigureRegion(
                            page=pno + 1,
                            bbox=(round(cluster.x0, 2), round(cluster.y0, 2), round(cluster.x1, 2), round(cluster.y1, 2)),
                            kind=kind,
                            area_ratio=round(ratio, 4),
                        )
                    )
        regions.sort(key=lambda r: (-r.area_ratio, r.page))
        return regions[:max_figures]
    except Exception as exc:
        tracer.event("tool_error", node="figure_tool", tool="detect_figure_regions", error=f"{type(exc).__name__}: {exc}")
        return []


def render_region_png(pdf_path: str, page: int, bbox: tuple[float, float, float, float],
                      dest_dir: str | Path | None = None) -> Path | None:
    """把 (page, bbox) 以 150 dpi 渲染为 PNG；默认存 data/indexes/figures/；失败返回 None。"""
    try:
        s = get_settings()
        out_dir = Path(dest_dir) if dest_dir is not None else s.indexes_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        key = sha256_text(f"{Path(pdf_path).resolve()}|{page}|{bbox}")[:8]
        out_path = out_dir / f"fig_p{page}_{key}.png"
        with pymupdf.open(pdf_path) as pdf:
            if not (1 <= page <= pdf.page_count):
                return None
            pg = pdf[page - 1]
            clip = pymupdf.Rect(*bbox) & pg.rect
            if clip.is_empty:
                return None
            pix = pg.get_pixmap(clip=clip, dpi=_RENDER_DPI)
            if not pix.width or not pix.height:
                return None
            pix.save(str(out_path))
        return out_path
    except Exception as exc:
        get_active_tracer().event(
            "tool_error", node="figure_tool", tool="render_region_png", error=f"{type(exc).__name__}: {exc}"
        )
        return None


def analyze_figure(pdf_path: str, page: int, bbox: tuple[float, float, float, float], question: str, *,
                   llm=None, figure_no: str | None = None) -> FigureReading | None:
    """VLM 结构化解读；vision 未启用/失败返回 None。figure_path/page 必须绑定真实值。"""
    tracer = get_active_tracer()
    try:
        s = get_settings()
        if not bool(s.source("vision", "enabled", False)) or llm is None:
            return None
        png = render_region_png(pdf_path, page, bbox)
        if png is None:
            tracer.event("tool_error", node="figure_tool", tool="analyze_figure", error="区域渲染失败")
            return None
        png_bytes = png.read_bytes()
        system = "你是论文图表阅读助手。只依据图像中直接可见的内容作答，不得臆造图中不存在的信息。"
        prompt = (
            f"这是论文 PDF 第 {page} 页的一个图像区域截图"
            + (f"（{figure_no}）" if figure_no else "")
            + "。\n"
            + f"需要回答的问题：{question or '这张图/表展示了什么？'}\n\n"
            "请输出结构化 JSON：\n"
            "description：只描述可从图中直接读出的内容（图形类型、坐标轴、趋势、构成），不要推断论文结论；\n"
            "numbers：从图中读出的关键数字（尽量带单位与所属条件）；\n"
            "caveats：看不清或不确定的点（模糊、遮挡、缺标签等）；\n"
            "modality：figure / table / formula 三选一；\n"
            "confidence：0 到 1 的解读置信度。"
        )
        reading = llm.chat_json(
            prompt, FigureReading, images=[(png_bytes, "image/png")], system=system, label="figure_tool:analyze"
        )
        # 强制绑定真实元数据，不信任模型输出的路径/页码
        reading.figure_path = str(png)
        reading.page = page
        if figure_no is not None:
            reading.figure_no = figure_no
        if reading.modality not in ("figure", "table", "formula"):
            # 模型偶回 "text"（deepseek-flash 实测）——图表解读绝不可能是 text：
            # EvidenceRecord 的 modality=="text" 意味着逐字 PDF 原文，VLM 描述冒充会破坏
            # Writer 的逐字契约与桩过滤语义
            reading.modality = "figure"
        tracer.event("figure_analyzed", node="figure_tool", page=page, figure=str(png), modality=reading.modality)
        return reading
    except Exception as exc:
        tracer.event("tool_error", node="figure_tool", tool="analyze_figure", error=f"{type(exc).__name__}: {exc}")
        return None
