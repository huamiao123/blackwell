"""Plot the six measured P0 blocks and the initial latency budget."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from g4_common import read_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('blocks', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rows = read_csv(args.blocks)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True, layout='constrained')
    for ax, order in zip(axes, ['ab', 'ba']):
        ax.axhline(0, color='#b7bdc4', linewidth=0.8)
        ax.axhline(5, color='#888888', linewidth=1, linestyle='--', label='5% initial budget')
        for precision, name, color, marker in [('bf16', 'BF16', '#284687', 'o'),
                                               ('nvfp4', 'NVFP4', '#a65716', 's')]:
            points = sorted((1 + int(row['block']), 100 * float(row['primary_interference']))
                            for row in rows if row['precision'] == precision and row['order'] == order)
            x, y = zip(*points)
            ax.plot(x, y, marker=marker, color=color, linewidth=1.3, label=name)
        ax.set_title(f'Queue order {order.upper()}')
        ax.set_xlabel('Matched block')
        ax.set_xticks(range(1, 7))
        ax.grid(axis='y', alpha=0.2)
    axes[0].set_ylabel('QKV event duration change (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=3, frameon=False, fontsize=8)
    fig.suptitle('QKV + FP32 Light, unprofiled GPU event envelopes')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
