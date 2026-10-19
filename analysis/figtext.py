"""图内文字体例与正文一致（软件学报：半角标点、中西文之间不留空格）；统一的保存入口。

导入即生效：给 matplotlib.text.Text.set_text 打补丁，所有坐标轴标签、标题、图例、注记在渲染前统一转换；
数学片段（$…$）与千分位逗号不动。规则与 tools/docx/jos_prep.py 的 transform_text 相同（顿号、引号、破折号保留）。

save(fig, path, **kw)：PNG 按 FIG_DPI（默认 600）保存；若设置了环境变量 FIG_VECTOR_DIR，同时在该目录写出
同名 .pdf / .svg（矢量，字体以 TrueType 子集嵌入）与 .tif（LZW，600 dpi），供期刊排版使用。
"""
import os
import re
from pathlib import Path

import matplotlib
import matplotlib.text

matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType 子集而非 Type 3，期刊可编辑/可检索
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"  # SVG 保留文字对象（需系统有同名字体；排版时亦可转曲）


def save(fig, path, **kw):
    path = Path(path)
    dpi = int(os.environ.get("FIG_DPI", "600"))
    fig.savefig(path, dpi=dpi, **kw)
    vd = os.environ.get("FIG_VECTOR_DIR")
    if vd:
        vd = Path(vd)
        vd.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg"):
            fig.savefig(vd / f"{path.stem}.{ext}", **kw)
        try:
            from PIL import Image
            with Image.open(path) as im:
                im.save(vd / f"{path.stem}.tif", compression="tiff_lzw", dpi=(dpi, dpi))
            im2 = Image.open(path); im2.save(vd / f"{path.stem}.png", dpi=(dpi, dpi)); im2.close()
        except ImportError:
            pass

PUNCT = str.maketrans({"，": ",", "。": ".", "；": ";", "：": ":", "（": "(", "）": ")", "？": "?", "！": "!"})
CJK = "\u4e00-\u9fff\u3000-\u303f\uff00-\uffef“”‘’—…·"
SPACE_AFTER_CJK = re.compile(rf"(?<=[{CJK}])[ \u00a0]+(?=[A-Za-z0-9αβγδεθκλμρσΔΣ\[\$])")
SPACE_BEFORE_CJK = re.compile(rf"(?<=[A-Za-z0-9%\]\$αβγδεθκλμρσΔΣ])[ \u00a0]+(?=[{CJK}])")
MATH = re.compile(r"(\$[^$]*\$)")


def jos_text(s):
    if not isinstance(s, str) or not s:
        return s
    out = []
    for i, part in enumerate(MATH.split(s)):
        if i % 2 == 0:
            part = part.translate(PUNCT)
            part = SPACE_AFTER_CJK.sub("", part)
            part = SPACE_BEFORE_CJK.sub("", part)
        out.append(part)
    return "".join(out)


_orig_set_text = matplotlib.text.Text.set_text


def _set_text(self, s):
    return _orig_set_text(self, jos_text(s))


matplotlib.text.Text.set_text = _set_text
