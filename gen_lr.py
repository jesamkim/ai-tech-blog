#!/usr/bin/env python3
"""Learning Rate Schedule - Cosine with Warmup"""
import matplotlib
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
fm._load_fontmanager(try_read_cache=False)

BG = "#1a1a2e"
TX = "#e0e0e0"
BLUE = "#4fc3f7"
GREEN = "#66bb6a"
YELLOW = "#ffd54f"
RED = "#ef5350"

fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
ax.set_facecolor(BG)

# Cosine schedule with warmup
max_steps = 1000
warmup_steps = 150  # 15%
max_lr = 1e-5
min_lr = 1e-6

steps = np.arange(max_steps)
lr = np.zeros(max_steps)

# Warmup phase (linear)
for i in range(warmup_steps):
    lr[i] = min_lr + (max_lr - min_lr) * (i / warmup_steps)

# Cosine decay phase
for i in range(warmup_steps, max_steps):
    progress = (i - warmup_steps) / (max_steps - warmup_steps)
    lr[i] = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))

# Plot
ax.plot(steps[:warmup_steps], lr[:warmup_steps], color=GREEN, linewidth=2.5, label="Warmup (Linear)")
ax.plot(steps[warmup_steps:], lr[warmup_steps:], color=BLUE, linewidth=2.5, label="Cosine Decay")

# Annotations
ax.axvline(x=warmup_steps, color=YELLOW, linestyle="--", alpha=0.5, linewidth=1)
ax.text(warmup_steps + 10, max_lr * 0.95, f"Warmup 종료\n({warmup_steps} steps, 15%)", 
        fontsize=9, color=YELLOW, alpha=0.8)

ax.axhline(y=max_lr, color=TX, linestyle=":", alpha=0.3)
ax.text(max_steps - 100, max_lr * 1.05, f"max_lr = {max_lr}", fontsize=9, color=TX, alpha=0.6)

ax.axhline(y=min_lr, color=TX, linestyle=":", alpha=0.3)
ax.text(max_steps - 100, min_lr * 1.3, f"min_lr = {min_lr}", fontsize=9, color=TX, alpha=0.6)

# Labels
ax.set_xlabel("Training Steps", fontsize=11, color=TX)
ax.set_ylabel("Learning Rate", fontsize=11, color=TX)
ax.set_title("Nova Forge Learning Rate Schedule (Cosine with Warmup)", 
             fontsize=14, color=TX, weight="bold", pad=15)

# Style
ax.tick_params(colors=TX, labelsize=9)
ax.spines["bottom"].set_color("#555555")
ax.spines["left"].set_color("#555555")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10, loc="upper right", facecolor=BG, edgecolor="#555555", labelcolor=TX)

# Watermark
ax.text(0.02, 0.02, "jesamkim.github.io", transform=ax.transAxes, fontsize=8, color=TX, alpha=0.3)

plt.tight_layout()
plt.savefig("static/images/lr-schedule-nova-forge.png", dpi=120, facecolor=BG, bbox_inches="tight")
print("Done: lr-schedule")
plt.close()
