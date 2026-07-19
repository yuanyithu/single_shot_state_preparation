import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .scan_results import load_exp102_publication_q_top


def plot_results(results_path, output_dir):
    publication = load_exp102_publication_q_top(results_path)
    output_dir = Path(output_dir)
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(publication["m_values"])))
    for index, (m, color) in enumerate(zip(publication["m_values"], colors)):
        axis.errorbar(publication["p_values"], publication["q_top"][index],
                      yerr=publication["errorbar"][index], marker="o", capsize=2.5,
                      linewidth=1.5, color=color, label=f"m={int(m)}")
    axis.set(xlabel="Data-error probability p", ylabel=r"$q_{top}$")
    axis.grid(alpha=0.22); axis.legend(ncol=2, frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"qtop_vs_p.{suffix}", dpi=220)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("results"); parser.add_argument("output_dir")
    args = parser.parse_args(argv); plot_results(args.results, args.output_dir)


if __name__ == "__main__": main()
