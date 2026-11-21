"""Render figures at their intended printed sizes for the review draft."""
from pathlib import Path
import runpy
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

project = Path(__file__).resolve().parents[2]
script = project / "work" / "practice.py"
output = project / "outputs"
sys.argv = [str(script), "--no-show"]
with threadpool_limits(limits=1):
    data = runpy.run_path(str(script), run_name="__main__")
plt.close("all")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

indices = np.arange(1, len(data["s"]) + 1)
fig, ax = plt.subplots(figsize=(3.2, 2.6))
ax.semilogy(indices, data["s"], label="Batch")
ax.semilogy(indices, data["s_inc"], "--", label="Incremental")
ax.set(xlabel="Singular-value index", ylabel="Singular value")
ax.legend()
fig.tight_layout(pad=0.4)
fig.savefig(output / "report_spectrum.png", dpi=300)

comparison = data["comparison"]
fig, ax = plt.subplots(figsize=(3.2, 2.6))
ax.plot(data["ranks"], comparison[:, 1], "o-", label="Batch")
ax.plot(data["ranks"], comparison[:, 2], "s--", label="Incremental")
ax.set(xlabel="Retained rank q", ylabel="Relative Frobenius error")
ax.set_xticks(data["ranks"])
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout(pad=0.4)
fig.savefig(output / "report_error.png", dpi=300)

j = data["frame_index"]
fig, axes = plt.subplots(1, 4, figsize=(6.8, 3.5))
images = [data["Y"][:, j].reshape(96, 54)]
images.extend(data["incremental_images"][q] for q in data["ranks"])
titles = ["Original"] + [f"Incremental\nrank {q}" for q in data["ranks"]]
for ax, image, title in zip(axes, images, titles):
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Column (pixels)")
    ax.set_xticks([0, 25, 50])
    ax.set_yticks([0, 30, 60, 90])
axes[0].set_ylabel("Row (pixels)")
for ax in axes[1:]:
    ax.tick_params(labelleft=False)
fig.suptitle(f"Sampled frame {j + 1} of {data['Y'].shape[1]}", fontsize=11)
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.18, top=0.78, wspace=0.15)
fig.savefig(output / "report_frames.png", dpi=300)
plt.close("all")
print("Report-sized figures rendered.")
