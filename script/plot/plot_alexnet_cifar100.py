import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ModelStyle:
    key: str
    label: str
    color: str
    marker: str
    dx: float
    dy: float


MODEL_STYLES = (
    ModelStyle("alexnet", "AlexNet", "black", "o", 1.8, 0.18),
    ModelStyle("tcl1", "TCL1", "blue", "s", 1.8, 0.18),
    ModelStyle("tcl12", "TCL12", "blue", "D", -13.0, -0.35),
    ModelStyle("qtcl1", "QTCL1", "red", "^", 1.8, -0.35),
    ModelStyle("qtcl12", "QTCL12", "red", "v", -13.0, 0.18),
)


def parse_args() -> argparse.Namespace:
    default_tex_path = Path(__file__).with_name("alex.tex")
    parser = argparse.ArgumentParser(description="Plot the AlexNet CIFAR-100 accuracy/space-saving trade-off.")
    parser.add_argument(
        "--tex-path",
        type=Path,
        default=default_tex_path,
        help="Path to the LaTeX table file containing the experiment values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the PNG and PDF figures are written. Defaults to the TeX file directory.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="alexnet_tradeoff",
        help="Output filename stem used for both PNG and PDF.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    return parser.parse_args()


def strip_latex_markup(text: str) -> str:
    cleaned = text.strip()
    bold_pattern = re.compile(r"\\textbf\{([^{}]+)\}")
    while True:
        updated = bold_pattern.sub(r"\1", cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = cleaned.replace("{", "").replace("}", "")
    return " ".join(cleaned.split())


def normalize_model_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", strip_latex_markup(text).lower())


def parse_percentage(value: str) -> float:
    cleaned = strip_latex_markup(value).replace(r"\%", "").replace("%", "").strip()
    if cleaned == "---":
        return 0.0
    return float(cleaned)


def parse_tradeoff_points(tex_path: Path) -> dict[str, dict[str, float]]:
    text = tex_path.read_text()
    parsed: dict[str, dict[str, float]] = {}
    expected_keys = {style.key for style in MODEL_STYLES}

    skip_prefixes = (
        "\\begin",
        "\\end",
        "\\centering",
        "\\small",
        "\\caption",
        "\\label",
        "\\setlength",
        "\\renewcommand",
        "\\toprule",
        "\\midrule",
        "\\bottomrule",
    )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "&" not in line:
            continue
        if line.startswith("%") or line.startswith(skip_prefixes):
            continue

        fields = [field.strip().rstrip("\\").strip() for field in line.split("&")]
        if len(fields) != 4:
            continue

        model_key = normalize_model_name(fields[0])
        if model_key not in expected_keys:
            continue

        parsed[model_key] = {
            "val_acc": parse_percentage(fields[1]),
            "space_saving": parse_percentage(fields[3]),
        }

    missing = [style.label for style in MODEL_STYLES if style.key not in parsed]
    if missing:
        raise ValueError(f"Missing rows in {tex_path}: {', '.join(missing)}")

    return parsed


def plot_tradeoff(points: dict[str, dict[str, float]], output_png: Path, output_pdf: Path, title: str | None) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(3, 2.5), dpi=1000)

    x_values = []
    y_values = []
    for style in MODEL_STYLES:
        point = points[style.key]
        x_value = point["space_saving"]
        y_value = point["val_acc"]
        x_values.append(x_value)
        y_values.append(y_value)
        ax.scatter(
            x_value,
            y_value,
            color=style.color,
            marker=style.marker,
            s=28,
            linewidths=0.8,
            edgecolors=style.color,
            label=style.label,
            zorder=3,
        )
        ax.annotate(
            style.label,
            (x_value, y_value),
            xytext=(x_value + style.dx, y_value + style.dy),
            textcoords="data",
            fontsize=9,
            color=style.color,
        )

    ax.set_xlabel("Space saving (%)")
    ax.set_ylabel("Validation accuracy (%)")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    y_margin = max(0.6, 0.12 * (y_max - y_min))

    ax.set_xlim(min(-2.0, x_min - 2.0), x_max + 2.5)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    fig.tight_layout()
    fig.savefig(output_png, dpi=1000, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=1000, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tex_path = args.tex_path.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else tex_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    points = parse_tradeoff_points(tex_path)
    output_png = output_dir / f"{args.stem}.png"
    output_pdf = output_dir / f"{args.stem}.pdf"
    plot_tradeoff(points, output_png, output_pdf, args.title)

    print(f"Source: {tex_path}")
    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")


if __name__ == "__main__":
    main()
