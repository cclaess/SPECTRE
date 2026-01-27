import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

if __name__ == "__main__":

    # Percentage of text dropped
    x = np.array([0, 10, 25, 50])

    # Token-wise dropout
    r1_token = [44.81, 42.93, 37.86, 23.57]
    r8_token = [84.30, 83.40, 79.04, 62.85]

    # Span-wise dropout
    r1_span = [44.81, 15.91, 4.12, 1.21]
    r8_span = [84.30, 49.52, 20.94, 9.23]

    # 95% CI bounds (fill these with your lower/upper values)
    r1_token_ci = ([43.82, 42.13, 37.13, 22.83], [45.82, 43.88, 38.55, 24.19])
    r8_token_ci = ([83.86, 83.04, 78.58, 62.34], [84.70, 83.79, 79.43, 63.33])
    r1_span_ci = ([43.82, 15.43, 3.79, 1.01], [45.82, 16.49, 4.44, 1.40])
    r8_span_ci = ([83.86, 49.07, 20.54, 8.91], [84.70, 50.02, 21.35, 9.52])

    random_r1 = 0.8
    random_r8 = 6.3

    plt.figure(figsize=(8, 5))

    # Token-wise
    plt.plot(
        x, r1_token, marker="o", linewidth=2, color="#387896",
        label="_nolegend_"
    )
    plt.plot(
        x, r8_token, marker="s", linewidth=2, color="#E97132",
        label="_nolegend_"
    )

    # Span-wise
    plt.plot(
        x, r1_span, marker="o", linestyle="--", linewidth=2, color="#387896",
        label="_nolegend_"
    )
    plt.plot(
        x, r8_span, marker="s", linestyle="--", linewidth=2, color="#E97132",
        label="_nolegend_"
    )

    # Shaded 95% CI bands (transparent fill between lower and upper bounds)
    plt.fill_between(x, r1_token_ci[0], r1_token_ci[1], color="#387896", alpha=0.15)
    plt.fill_between(x, r8_token_ci[0], r8_token_ci[1], color="#E97132", alpha=0.15)
    plt.fill_between(x, r1_span_ci[0], r1_span_ci[1], color="#387896", alpha=0.10)
    plt.fill_between(x, r8_span_ci[0], r8_span_ci[1], color="#E97132", alpha=0.10)

    # Random chance lines
    plt.axhline(
        random_r1, linestyle="--", linewidth=1.5, color="black",
        label="_nolegend_"
    )
    plt.axhline(
        random_r8, linestyle="--", linewidth=1.5, color="black",
        label="_nolegend_"
    )

    # Annotate random chance lines just above the lines (left side of plot)
    plt.text(x[0], random_r1 + 0.8, "Random chance R@1", ha="left", va="bottom", color="black")
    plt.text(x[0], random_r8 + 0.8, "Random chance R@8", ha="left", va="bottom", color="black")

    plt.xlabel("Dropped text tokens (%)", fontsize=12)
    plt.ylabel("Recall", fontsize=12)
    plt.xticks(x)
    plt.ylim(bottom=0.0)
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    legend_handles = [
        Line2D([0], [0], color="#387896", linewidth=2, label="Recall@1"),
        Line2D([0], [0], color="#E97132", linewidth=2, label="Recall@8"),
        Line2D([0], [0], color="gray", linewidth=2, linestyle="-", label="token dropout"),
        Line2D([0], [0], color="gray", linewidth=2, linestyle="--", label="span dropout"),
    ]
    plt.legend(handles=legend_handles, frameon=False, ncol=2, loc="upper right", fontsize=12)
    plt.tight_layout()

    plt.savefig("retrieval_degradation_merlin.png", dpi=300)
    plt.savefig("retrieval_degradation_merlin.pdf")

    plt.show()
