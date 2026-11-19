"""Show P2 GPU execution windows for QKV and Gate/Up under AB submission."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from g4_common import read_csv


def interval_pair(path, case):
    rows = {r['role']: r for r in read_csv(path) if r['mode'] == 'parallel_ab' and r['case'] == case}
    anchor = min(int(row['start_ns']) for row in rows.values())
    return {role: tuple((int(row[key]) - anchor) / 1000 for key in ('start_ns', 'end_ns'))
            for role, row in rows.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), layout='constrained')
    panels = [
        (axes[0], 'QKV projection', 'qkv_proj', [('FP32 Light', 'grid64_trace'),
            ('FP32 Heavy', 'grid64_fp32_heavy_trace'), ('BF16 MMA', 'grid64_tensor_light_trace')]),
        (axes[1], 'Gate/Up projection', 'gate_up_proj', [('FP32 Light', 'gate_up_grid64_trace')]),
    ]
    for ax, title, operator, groups in panels:
        labels = []
        positions = []
        for index, (probe, directory) in enumerate(groups):
            path = args.analysis / directory / 'qkv_pilot_intervals.csv'
            for precision_index, precision in enumerate(('nvfp4', 'bf16')):
                position = 2.5 * index + precision_index
                labels.append(f'{precision.upper()} · {probe}')
                positions.append(position)
                pair = interval_pair(path, f'{precision}_{operator}')
                a_start, a_end = pair['A']
                b_start, b_end = pair['B']
                ax.broken_barh([(a_start, a_end - a_start)], (position - 0.32, 0.29), facecolor='#2a5788')
                ax.broken_barh([(b_start, b_end - b_start)], (position + 0.03, 0.29), facecolor='#d88b2a')
                tail = max(a_end, b_start)
                if b_end > tail:
                    ax.broken_barh([(tail, b_end - tail)], (position + 0.03, 0.29),
                                   facecolor='#edd0aa', edgecolor='#965225', hatch='///', linewidth=0.45)
        ax.set_yticks(positions, labels, fontsize=8)
        ax.set_ylim(-0.54, positions[-1] + 0.54)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('Time since first GPU kernel start (us)')
        ax.grid(axis='x', alpha=0.2)
    fig.suptitle('64-CTA probes, single P2 AB trace per configuration')
    fig.legend(handles=[Patch(color='#2a5788', label='Projection'), Patch(color='#d88b2a', label='Probe'),
                        Patch(facecolor='#edd0aa', edgecolor='#965225', hatch='///', label='Probe after projection')],
               loc='outside lower center', ncols=3, frameon=False, fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == '__main__':
    main()
