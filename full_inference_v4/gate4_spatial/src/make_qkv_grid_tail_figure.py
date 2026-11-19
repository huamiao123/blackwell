"""Show the measured GPU overlap and the FP32 work's tail in the P2 AB traces."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from g4_common import read_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grids', type=int, nargs='+', default=[128, 32])
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 2.5 + 0.9 * len(args.grids)), sharex=True, layout='constrained')
    for ax, precision in zip(axes, ['nvfp4', 'bf16']):
        for y, grid in enumerate(args.grids):
            path = args.analysis / f'grid{grid}_trace/qkv_pilot_intervals.csv'
            rows = {r['role']: r for r in read_csv(path)
                    if r['case'] == precision + '_qkv_proj' and r['mode'] == 'parallel_ab'}
            origin = min(int(row['start_ns']) for row in rows.values())
            a_begin, a_end = ((int(rows['A'][name]) - origin) / 1000 for name in ['start_ns', 'end_ns'])
            b_begin, b_end = ((int(rows['B'][name]) - origin) / 1000 for name in ['start_ns', 'end_ns'])
            a_tail = max(a_end, b_begin)
            ax.broken_barh([(a_begin, a_end - a_begin)], (y - 0.30, 0.25), facecolor='#2a5788')
            ax.broken_barh([(b_begin, b_end - b_begin)], (y + 0.03, 0.25), facecolor='#d88b2a')
            if b_end > a_tail:
                ax.broken_barh([(a_tail, b_end - a_tail)], (y + 0.03, 0.25),
                               facecolor='#edd0aa', edgecolor='#965225', hatch='///', linewidth=0.45)
                ax.text(b_end + 0.8, y + 0.15, f'{b_end - a_tail:.1f} us tail', va='center', fontsize=8)
        ax.set_yticks(range(len(args.grids)), [f'{grid} CTA' for grid in args.grids])
        ax.set_title(f'{precision.upper()} QKV with FP32 Light', loc='left', fontsize=11)
        ax.set_ylim(-0.45, len(args.grids) - 0.54)
        ax.grid(axis='x', alpha=0.2)
    axes[-1].set_xlabel('Time from first GPU kernel start (us)')
    axes[-1].set_xlim(0, 83)
    fig.suptitle('Single P2 AB trace per condition, same total FP32 work')
    fig.legend(handles=[Patch(color='#2a5788', label='QKV'), Patch(color='#d88b2a', label='FP32 probe'),
                        Patch(facecolor='#edd0aa', edgecolor='#965225', hatch='///', label='Probe after QKV')],
               loc='outside lower center', ncols=3, frameon=False, fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
