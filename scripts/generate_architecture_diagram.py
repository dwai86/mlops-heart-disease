from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

root = Path(__file__).resolve().parents[1]
out_path = root / "screenshots" / "architecture_diagram.png"
out_path.parent.mkdir(exist_ok=True, parents=True)

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

nodes = {
    "client": (1.4, 6.5),
    "api": (4.8, 6.5),
    "model": (8.0, 6.5),
    "mlflow": (10.0, 5.1),
    "ci": (8.0, 3.5),
    "prom": (5.0, 2.0),
    "kube": (9.0, 1.2),
}

for name, (x, y) in nodes.items():
    if name == "client":
        patch = Circle((x, y), 0.9, color="#dceefb", ec="black", lw=1.5)
    elif name in {"api", "model", "mlflow", "ci", "prom"}:
        patch = FancyBboxPatch(
            (x - 1.2, y - 0.7),
            2.4,
            1.4,
            boxstyle="round,pad=0.1",
            color={
                "api": "#cde7d4",
                "model": "#f8e4b5",
                "mlflow": "#f3d6d6",
                "ci": "#e6d9f7",
                "prom": "#dcc2a5",
            }[name],
            ec="black",
            lw=1.5,
        )
    else:
        patch = FancyBboxPatch(
            (x - 1.6, y - 0.7),
            3.2,
            1.4,
            boxstyle="round,pad=0.1",
            color="#dcdcdc",
            ec="black",
            lw=1.5,
        )
    ax.add_patch(patch)
    ax.text(x, y, name.replace("_", " ").title(), ha="center", va="center", fontsize=10)

arrow_specs = [
    ("client", "api", "HTTP requests", 0.5),
    ("api", "model", "Prediction request", 0.5),
    ("model", "mlflow", "Logs + metrics", 0.5),
    ("model", "ci", "Training + checks", 0.6),
    ("api", "prom", "/metrics", 0.6),
    ("api", "kube", "Deployable service", 0.5),
]

for source, target, label, offset in arrow_specs:
    x1, y1 = nodes[source]
    x2, y2 = nodes[target]
    if source == "client":
        start = (x1 + 0.8, y1)
        end = (x2 - 1.2, y2)
    elif source == "api" and target == "model":
        start = (x1 + 1.2, y1)
        end = (x2 - 1.2, y2)
    elif source == "model" and target == "mlflow":
        start = (x1 + 1.0, y1 + 0.1)
        end = (x2 - 0.8, y2 + 0.2)
    elif source == "model" and target == "ci":
        start = (x1 + 0.1, y1 - 0.7)
        end = (x2 + 0.1, y2 + 0.7)
    elif source == "api" and target == "prom":
        start = (x1 + 0.5, y1 - 0.7)
        end = (x2 + 0.9, y2 + 0.4)
    else:
        start = (x1 + 0.8, y1 - 0.6)
        end = (x2 - 0.8, y2 + 0.5)
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"),
    )
    x_mid = (start[0] + end[0]) / 2
    y_mid = (start[1] + end[1]) / 2 + offset
    ax.text(x_mid, y_mid, label, fontsize=8, ha="center", color="#333333")

fig.tight_layout()
fig.savefig(out_path, dpi=200)
print(f"Saved architecture diagram to {out_path}")
