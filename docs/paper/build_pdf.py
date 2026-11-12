"""Build the H-net Paper Track draft: figure data, figures, word count, and the PDF.

Usage (from the repository root, with the paper venv):
    python docs/paper/build_pdf.py --extract ~/wt-P3.9b/runs/P3.9-02/run_logs/arms
    python docs/paper/build_pdf.py            # figures from figures/fig2_loss.csv, then the PDF
    python docs/paper/build_pdf.py --count    # word count only

Why: the machine has no LaTeX, pandoc or browser, so the PDF is drawn with reportlab Platypus
from the Markdown source, which keeps LaTeX math and [@key] citations resolved against refs.bib.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE / "h-net-draft.md"
BIB = HERE / "refs.bib"
FIG = HERE / "figures"
PDF = HERE / "h-net-draft.pdf"
CSV = FIG / "fig2_loss.csv"
ARMS = {"c": "sgd", "f": "h-fit", "g": "h-merge-fit"}
LABELS = {"c": "(c) plain TTT", "f": "(f) TTT from H's factors", "g": "(g) merge H, then TTT"}
# Categorical slots 1 to 3 of the dataviz reference palette, validated for CVD separation.
COLORS = {"c": "#2a78d6", "f": "#eb6834", "g": "#1baf7a"}
STYLES = {"c": "-", "f": "--", "g": "-."}
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


# --------------------------------------------------------------------------- figure data


def trailing(values: list[float], window: int = 8) -> list[float]:
    """Trailing mean over min(window, s) steps, the P3.9-02 report's smoothing."""
    out = []
    for s in range(1, len(values) + 1):
        w = min(window, s)
        out.append(sum(values[s - w : s]) / w)
    return out


def extract(arms_dir: Path) -> None:
    """Write figures/fig2_loss.csv from P3.9-02's records-rank*.jsonl and print the checks.

    Args:
        arms_dir: the run's ``run_logs/arms`` folder holding ``records-rank*.jsonl``.
    """
    recs = []
    for f in sorted(glob.glob(str(arms_dir / "records-rank*.jsonl"))):
        recs += [json.loads(line) for line in open(f)]
    mean = {}
    for key, arm in ARMS.items():
        rows = [r for r in recs if r["arm"] == arm and r["status"] == "done"]
        if len(rows) != 40:
            raise SystemExit(f"{arm}: {len(rows)} done records, expected 40")
        for r in rows:
            if [e["step"] for e in r["train_log"]] != list(range(1, 129)):
                raise SystemExit(f"{arm} {r['puzzle_id']}: train_log is not steps 1 to 128")
        mean[key] = [sum(r["train_log"][s]["loss"] for r in rows) / len(rows) for s in range(128)]
    trail = {k: trailing(v) for k, v in mean.items()}
    FIG.mkdir(exist_ok=True)
    with open(CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step"] + [f"mean_{k}" for k in "cfg"] + [f"trailing8_{k}" for k in "cfg"])
        for s in range(128):
            w.writerow(
                [s + 1]
                + [f"{mean[k][s]:.8f}" for k in "cfg"]
                + [f"{trail[k][s]:.8f}" for k in "cfg"]
            )
    below = [trail["g"][i] < trail["c"][i] for i in range(128)]
    lead = next(i for i, b in enumerate(below) if not b)
    crossings = sum(below[i] != below[i - 1] for i in range(1, 128))
    print(f"wrote {CSV}")
    print(f"(g) below (c) through step {lead}; crossings {crossings}; below on {sum(below)} steps")
    print("finals", {k: round(trail[k][-1], 4) for k in "cfg"})
    print("step 1", {k: round(mean[k][0], 4) for k in "cfg"})


def read_curves() -> dict[str, list[float]]:
    """Read the trailing-mean curves from figures/fig2_loss.csv."""
    with open(CSV) as fh:
        rows = list(csv.DictReader(fh))
    return {k: [float(r[f"trailing8_{k}"]) for r in rows] for k in "cfg"}


# --------------------------------------------------------------------------- figures


def fig_pipeline() -> None:
    """Figure 1: the merge-then-train pipeline, two rows of boxes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(7.0, 2.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 37)
    ax.axis("off")
    ink, muted, new_fill, old_fill = "#1f1f1f", "#5c5c58", "#dcebfb", "#f4f4f1"
    w, h = 21.0, 11.0
    top = [
        (2, "Demonstration\npairs", "input grids, outputs", False),
        (27, "Frozen base\nas encoder", "layer-24 mean pool", True),
        (52, "Hypernetwork H", "135M parameters", True),
        (77, "Rank-16 LoRA", "q, k, v, o; 36 layers", True),
    ]
    bottom = [
        (77, "Merge into\nbase weights", "bf16; restored after", True),
        (52, "Standard TTT", "fresh LoRA from B = 0", False),
        (27, "Sampling and\nrescoring", "two attempts", False),
    ]

    def box(x: float, y: float, title: str, sub: str, new: bool) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                linewidth=1.0, edgecolor=ink if new else muted,
                facecolor=new_fill if new else old_fill,
            )
        )
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=8.6,
                color=ink, fontweight="bold", linespacing=1.05)
        ax.text(x + w / 2, y + h * 0.17, sub, ha="center", va="center", fontsize=7.0, color=muted)

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                     linewidth=1.0, color=ink, shrinkA=0, shrinkB=0))

    yt, yb = 23.5, 3.5
    for x, t, s, n in top:
        box(x, yt, t, s, n)
    for x, t, s, n in bottom:
        box(x, yb, t, s, n)
    for (x0, *_), (x1, *_) in zip(top, top[1:], strict=False):
        arrow(x0 + w + 0.6, yt + h / 2, x1 - 0.6, yt + h / 2)
    arrow(77 + w / 2, yt - 0.6, 77 + w / 2, yb + h + 0.6)
    for (x0, *_), (x1, *_) in zip(bottom, bottom[1:], strict=False):
        arrow(x0 - 0.6, yb + h / 2, x1 + w + 0.6, yb + h / 2)
    ax.add_patch(FancyBboxPatch((2.6, yb + h / 2 - 1.3), 3.2, 2.6, boxstyle="round,pad=0.2",
                                facecolor=new_fill, edgecolor=ink, linewidth=0.8))
    ax.text(7.6, yb + h / 2, "new: 2.3 s\nper task", fontsize=7.4, color=muted,
            va="center", ha="left", linespacing=1.15)
    fig.savefig(FIG / "fig1_pipeline.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG / "fig1_pipeline.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_loss() -> None:
    """Figure 2: trailing-mean training loss of (c), (f), (g) and each curve over (c)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = read_curves()
    steps = list(range(1, 129))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(6.4, 4.3), sharex=True, gridspec_kw={"height_ratios": [2.1, 1]}
    )
    grid, muted = "#e4e4df", "#5c5c58"
    for a in (ax, axr):
        a.grid(True, color=grid, linewidth=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(muted)
        a.tick_params(colors=muted, labelcolor="#1f1f1f")
        a.axvline(45, color=muted, linestyle=":", linewidth=1.0)
    for k in "cfg":
        ax.plot(steps, curves[k], STYLES[k], color=COLORS[k], linewidth=1.8, label=LABELS[k])
    ax.set_yscale("log")
    ax.set_ylabel("training loss (log scale)")
    ax.set_yticks([0.01, 0.02, 0.05, 0.1])
    ax.set_yticklabels(["0.01", "0.02", "0.05", "0.1"])
    ax.legend(frameon=False, loc="lower left", fontsize=7.6)
    # Direct end labels; (g) is nudged up because its final value sits next to (c)'s.
    for k, dy in (("c", 0), ("f", 0), ("g", 7)):
        ax.annotate(f"{k}  {curves[k][-1]:.4f}", (128, curves[k][-1]), xytext=(4, dy),
                    textcoords="offset points", fontsize=7.4, color="#1f1f1f", va="center",
                    ha="left")
    ax.text(46.5, 0.075, "step 45", fontsize=7.4, color=muted)
    axr.axhline(1.0, color=COLORS["c"], linewidth=1.4)
    ratio = [curves["g"][i] / curves["c"][i] for i in range(128)]
    axr.plot(steps, ratio, STYLES["g"], color=COLORS["g"], linewidth=1.8)
    axr.set_ylim(0.9, 1.1)
    axr.set_yticks([0.9, 0.95, 1.0, 1.05, 1.1])
    axr.text(3, 1.065, "(g) above (c)", fontsize=7.4, color=muted)
    axr.text(3, 0.915, "(g) below (c)", fontsize=7.4, color=muted)
    axr.set_ylabel("(g) / (c)")
    axr.set_xlabel("TTT step")
    axr.set_xlim(1, 128)
    axr.set_xticks([1, 16, 32, 45, 64, 80, 96, 112, 128])
    fig.align_ylabels((ax, axr))
    fig.tight_layout(h_pad=0.6)
    fig.savefig(FIG / "fig2_loss.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG / "fig2_loss.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# --------------------------------------------------------------------------- markdown model


def parse_md(text: str) -> list[tuple[str, object]]:
    """Split the Markdown source into blocks.

    Returns:
        A list of (kind, payload): ``title``, ``subtitle``, ``h2``, ``para``, ``bullets``,
        ``image`` (path, caption), ``table`` (rows, caption), ``references``.
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    blocks: list[tuple[str, object]] = []
    for chunk in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.rstrip() for ln in chunk.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        first = lines[0]
        if first.startswith("# "):
            blocks.append(("title", first[2:].strip()))
        elif first.startswith("> "):
            blocks.append(("subtitle", " ".join(ln.lstrip("> ") for ln in lines)))
        elif first.startswith("## "):
            name = first[3:].strip()
            blocks.append(("references" if name == "References" else "h2", name))
        elif first.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.*)\)", " ".join(lines), re.S)
            blocks.append(("image", (m.group(2), m.group(1))))
        elif first.startswith("|"):
            rows = [
                [c.strip() for c in ln.strip("|").split("|")]
                for ln in lines
                if ln.startswith("|") and not re.match(r"^\|[-:| ]+\|$", ln)
            ]
            cap = " ".join(ln[2:] for ln in lines if ln.startswith(": "))
            blocks.append(("table", (rows, cap)))
        elif first.startswith(": ") and blocks and blocks[-1][0] == "table":
            rows, _ = blocks[-1][1]
            blocks[-1] = ("table", (rows, " ".join(ln[2:] for ln in lines)))
        elif first.startswith("- "):
            items, cur = [], []
            for ln in lines:
                if ln.startswith("- "):
                    if cur:
                        items.append(" ".join(cur))
                    cur = [ln[2:]]
                else:
                    cur.append(ln.strip())
            items.append(" ".join(cur))
            blocks.append(("bullets", items))
        else:
            blocks.append(("para", " ".join(lines)))
    return blocks


def counted_text(blocks: list[tuple[str, object]]) -> str:
    """The text under the 1,500-word limit: headings, body, bullets, table cells, captions."""
    parts = []
    for kind, payload in blocks:
        if kind in ("h2", "para"):
            parts.append(payload)
        elif kind == "bullets":
            parts += payload
        elif kind == "image":
            parts.append(payload[1])
        elif kind == "table":
            rows, cap = payload
            parts += [" ".join(r) for r in rows] + [cap]
    return "\n".join(parts)


def word_count(blocks: list[tuple[str, object]]) -> tuple[int, int]:
    """Whitespace tokens of the counted text, with math rendered and markup removed.

    Returns:
        (words, tokens): ``words`` skips tokens without a letter or digit (``=``, ``|``);
        ``tokens`` counts every whitespace-separated token.
    """
    plain = re.sub(r"<[^>]+>", "", md_inline(counted_text(blocks), cites={}, plain=True))
    plain = plain.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    toks = plain.split()
    return sum(1 for t in toks if re.search(r"[0-9A-Za-z\u0370-\u03ff\u00c0-\u024f]", t)), len(toks)


# --------------------------------------------------------------------------- inline rendering

GREEK = {r"\alpha": "α", r"\theta": "θ", r"\Delta": "Δ", r"\sum": "Σ", r"\times": "×",
         r"\in": "∈", r"\mid": "|", r"\log": "log ", r"\cdot": "·"}


def tex(expr: str) -> str:
    """Render the small LaTeX subset used in the draft as reportlab paragraph markup."""
    s = expr
    s = re.sub(r"\\mathbb\{R\}", "ℝ", s)
    s = re.sub(r"\\mathcal\{L\}", "<i>L</i>", s)
    s = re.sub(r"\\bar\{(\w)\}", lambda m: f"<i>{m.group(1)}\u0304</i>", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1/\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"√\1", s)
    for k in sorted(GREEK, key=len, reverse=True):
        s = s.replace(k, GREEK[k])
    s = re.sub(r"\^\{([^{}]*)\}", r"<super>\1</super>", s)
    s = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"_(\w)", r"<sub>\1</sub>", s)
    s = s.replace("\\,", " ").replace("\\;", " ").replace("\\ ", " ")
    s = re.sub(r"(?<![<\w/])([A-Za-z])(?![\w>])", r"<i>\1</i>", s)
    s = s.replace("-", "−").replace("'", "′")
    s = re.sub(r"<(/?)(super|sub|i)>", r"<\1\2>", s)
    return s


def md_inline(text: str, cites: dict[str, int], plain: bool = False) -> str:
    """Markdown inline markup, math and citations to reportlab markup."""
    maths: list[str] = []

    def stash(m: re.Match) -> str:
        maths.append(tex(m.group(1)))
        return f"\x00{len(maths) - 1}\x00"

    s = re.sub(r"\$([^$]+)\$", stash, text)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)

    def cite(m: re.Match) -> str:
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        if plain:
            return "[" + ", ".join("0" for _ in keys) + "]"
        nums = []
        for k in keys:
            if k not in cites:
                cites[k] = len(cites) + 1
            nums.append(str(cites[k]))
        return "[" + ", ".join(nums) + "]"

    s = re.sub(r"\[(@[^\]]+)\]", cite, s)
    s = re.sub(
        r"\[(PLACEHOLDER[^\]]*|CITATION NEEDED[^\]]*)\]",
        r'<font color="#b3261e"><b>[\1]</b></font>' if not plain else r"[\1]",
        s,
    )
    s = re.sub(r"\x00(\d+)\x00", lambda m: maths[int(m.group(1))], s)
    return s


# --------------------------------------------------------------------------- bibliography


def parse_bib(text: str) -> dict[str, dict[str, str]]:
    """A minimal BibTeX reader for the fetched entries in refs.bib."""
    entries: dict[str, dict[str, str]] = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),", text):
        start, depth, i = m.end(), 1, m.end()
        while depth and i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[start : i - 1]
        fields: dict[str, str] = {"_type": m.group(1).lower()}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            j, val = fm.end(), ""
            if j < len(body) and body[j] == "{":
                d, k = 1, j + 1
                while d and k < len(body):
                    d += {"{": 1, "}": -1}.get(body[k], 0)
                    k += 1
                val = body[j + 1 : k - 1]
            else:
                val = re.match(r"[^,]*", body[j:]).group(0)
            fields.setdefault(fm.group(1).lower(), val.strip())
        entries[m.group(2).strip()] = fields
    return entries


def bib_text(s: str) -> str:
    """Strip BibTeX braces and the few accent macros in refs.bib."""
    s = s.replace("{\\c{c}}", "ç").replace("\\c{c}", "ç").replace("\\_", "_")
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    return re.sub(r"[{}]", "", " ".join(s.split()))


def authors(raw: str) -> str:
    """'Last, First and ...' or 'First Last and ...' to 'First Last, ...'; et al. after six."""
    names = []
    for a in bib_text(raw).split(" and "):
        a = a.strip()
        if "," in a:
            last, first = [p.strip() for p in a.split(",", 1)]
            a = f"{first} {last}"
        names.append(a)
    if len(names) > 6:
        return names[0] + " et al."
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def reference_line(n: int, e: dict[str, str]) -> str:
    """One reference, plain style: authors. Title. Venue or arXiv id, year. URL."""
    title = bib_text(e.get("title", ""))
    year = bib_text(e.get("year", ""))
    if "eprint" in e:
        where = f"arXiv:{e['eprint']}, {year}."
    elif "booktitle" in e:
        doi = e.get("doi", "")
        where = f"In <i>{bib_text(e['booktitle'])}</i>, {year}." + (f" doi:{doi}" if doi else "")
    else:
        where = f"{bib_text(e.get('howpublished', ''))}, {year}. {bib_text(e.get('note', ''))}."
    who = authors(e.get("author", ""))
    s = f"[{n}] {who}{'' if who.endswith('.') else '.'} {title}. {where}"
    return s.replace("&", "&amp;").replace("&amp;lt;", "&lt;")


# --------------------------------------------------------------------------- PDF


def build_pdf() -> None:
    """Render the Markdown source, figures and references to h-net-draft.pdf."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Flowable,
        Image,
        KeepTogether,
        ListFlowable,
        ListItem,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    for name, file in {
        "Serif": "DejaVuSerifCondensed.ttf", "Serif-Bold": "DejaVuSerifCondensed-Bold.ttf",
        "Serif-Italic": "DejaVuSerifCondensed-Italic.ttf",
        "Serif-BoldItalic": "DejaVuSerifCondensed-BoldItalic.ttf",
        "Sans": "DejaVuSans.ttf", "Sans-Bold": "DejaVuSans-Bold.ttf",
    }.items():
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold",
                                  italic="Serif-Italic", boldItalic="Serif-BoldItalic")
    ink, muted = colors.HexColor("#1f1f1f"), colors.HexColor("#5c5c58")
    body = ParagraphStyle("body", fontName="Serif", fontSize=9.6, leading=13.4,
                          alignment=TA_JUSTIFY, textColor=ink, spaceAfter=5.5)
    styles = {
        "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=16.5, leading=20.5,
                                textColor=ink, spaceAfter=7, alignment=TA_LEFT),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontName="Serif-Italic",
                                   fontSize=10.4, leading=14.2, textColor=muted,
                                   alignment=TA_LEFT, spaceAfter=4),
        "note": ParagraphStyle("note", fontName="Sans", fontSize=7.6, leading=10,
                               textColor=muted, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.6, leading=15,
                             textColor=ink, spaceBefore=9, spaceAfter=4),
        "abstract": ParagraphStyle("abstract", parent=body, leftIndent=0.8 * cm,
                                   rightIndent=0.8 * cm, fontSize=9.4, leading=13),
        "caption": ParagraphStyle("caption", parent=body, fontSize=8.4, leading=11.4,
                                  textColor=ink, alignment=TA_LEFT, spaceBefore=3,
                                  spaceAfter=9),
        "cell": ParagraphStyle("cell", fontName="Serif", fontSize=8.8, leading=11,
                               textColor=ink),
        "ref": ParagraphStyle("ref", fontName="Serif", fontSize=8.2, leading=10.8,
                              textColor=ink, leftIndent=0.62 * cm, firstLineIndent=-0.62 * cm,
                              spaceAfter=2.5),
    }

    class Outline(Flowable):
        """Zero-size flowable that adds a PDF bookmark and an outline entry at its position."""

        def __init__(self, title: str, level: int) -> None:
            super().__init__()
            self.title, self.level = title, level
            self.key = f"o{id(self)}"

        def wrap(self, *_: float) -> tuple[float, float]:
            return 0, 0

        def draw(self) -> None:
            self.canv.bookmarkPage(self.key)
            self.canv.addOutlineEntry(self.title, self.key, level=self.level)

    text = MD.read_text()
    blocks = parse_md(text)
    words, tokens = word_count(blocks)
    cites: dict[str, int] = {}
    bib = parse_bib(BIB.read_text())
    story: list = []
    width = A4[0] - 4.2 * cm
    section = ""
    fig_n = 0
    for kind, payload in blocks:
        if kind == "title":
            story += [Outline(payload, 0), Paragraph(md_inline(payload, cites), styles["title"])]
        elif kind == "subtitle":
            story.append(Paragraph(md_inline(payload, cites), styles["subtitle"]))
            story.append(Paragraph(
                f"Draft 1, 4 November 2026. ARC Prize 2026 Paper Track. {words:,} words "
                "(limit 1,500, captions and table included; title, subtitle and references "
                "excluded). Red brackets mark placeholders.", styles["note"]))
        elif kind == "h2":
            section = payload
            story += [Outline(payload, 0), Paragraph(payload, styles["h2"])]
        elif kind == "para":
            m = re.match(r"\*\*(.+?)\*\*", payload)
            if m and section != "Abstract":
                story.append(Outline(re.sub(r"[.:]$", "", m.group(1)), 1))
            style = styles["abstract"] if section == "Abstract" else body
            story.append(Paragraph(md_inline(payload, cites), style))
        elif kind == "bullets":
            items = [ListItem(Paragraph(md_inline(t, cites), body), leftIndent=12, value="•")
                     for t in payload]
            story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=12,
                                      bulletFontName="Serif", bulletFontSize=9))
        elif kind == "image":
            path, cap = payload
            fig_n += 1
            img = Image(str(HERE / path))
            scale = (width * (0.98 if fig_n == 1 else 0.86)) / img.imageWidth
            img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
            cap_md = re.sub(r"^(Figure \d+\.)", r"**\1**", cap)
            story.append(KeepTogether([Spacer(1, 3), img,
                                       Paragraph(md_inline(cap_md, cites), styles["caption"])]))
        elif kind == "table":
            rows, cap = payload
            data = [[Paragraph(f"<b>{md_inline(c, cites)}</b>" if r == 0 else md_inline(c, cites),
                               ParagraphStyle(f"c{r}{j}", parent=styles["cell"],
                                              alignment=0 if j == 0 else 2))
                     for j, c in enumerate(row)] for r, row in enumerate(rows)]
            col = [width * 0.34] + [width * 0.22] * 3
            t = Table(data, colWidths=col, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("LINEABOVE", (0, 0), (-1, 0), 1.0, ink),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, ink),
                ("LINEBELOW", (0, -1), (-1, -1), 1.0, ink),
                ("TOPPADDING", (0, 0), (-1, -1), 2.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            cap_md = re.sub(r"^(Table \d+\.)", r"**\1**", cap)
            story.append(KeepTogether([Paragraph(md_inline(cap_md, cites), styles["caption"]),
                                       t, Spacer(1, 8)]))
        elif kind == "references":
            story += [Outline("References", 0), Paragraph("References", styles["h2"])]
            for key, n in sorted(cites.items(), key=lambda kv: kv[1]):
                if key not in bib:
                    raise SystemExit(f"citation {key} is not in refs.bib")
                story.append(Paragraph(reference_line(n, bib[key]), styles["ref"]))
            unused = sorted(set(bib) - set(cites))
            if unused:
                raise SystemExit(f"refs.bib holds uncited entries: {unused}")

    def footer(canv, doc) -> None:  # noqa: ANN001
        canv.saveState()
        canv.setFont("Sans", 7.5)
        canv.setFillColor(muted)
        canv.drawString(2.1 * cm, 1.2 * cm, "Demonstrations to Weights (draft 1)")
        canv.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, str(doc.page))
        canv.restoreState()

    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm,
                            topMargin=1.9 * cm, bottomMargin=2.0 * cm,
                            title="Demonstrations to Weights", author="arc26",
                            subject="ARC Prize 2026 Paper Track draft")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {PDF}: {words} words ({tokens} tokens), {len(cites)} references")


def main() -> None:
    """Parse arguments and run the requested steps."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extract", type=Path, help="P3.9-02 run_logs/arms folder to read")
    ap.add_argument("--count", action="store_true", help="print the word count and stop")
    args = ap.parse_args()
    if args.extract:
        extract(args.extract.expanduser())
    blocks = parse_md(MD.read_text())
    words, tokens = word_count(blocks)
    print(f"word count: {words} words, {tokens} whitespace tokens (limit 1,500)")
    if args.count:
        return
    fig_pipeline()
    fig_loss()
    build_pdf()


if __name__ == "__main__":
    main()
