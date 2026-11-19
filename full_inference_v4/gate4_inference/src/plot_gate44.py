"""Render paired request timing and phase-specific probe competition evidence."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis', type=Path)
    parser.add_argument('figure', type=Path)
    args = parser.parse_args()
    confidence = [row for path in sorted(args.analysis.glob('*/statistics.csv'))
                  for row in read(path) if row['metric'] == 'main_wall_change']
    traces = [row for row in read(args.analysis / 'trace_summary.csv')
              if row['phase_condition'] == row['actual_stage'].lower()]
    phases = [phase for phase in ('prefill', 'decode', 'lifecycle')
              if any(row['phase'] == phase for row in confidence)]
    trace_phases = [phase for phase in ('prefill', 'decode')
                    if any(row['phase_condition'] == phase for row in traces)]
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.1), layout='constrained')
    for label, color, shift in [('nvfp4', '#a95326', -0.15), ('bf16', '#275986', 0.15)]:
        data = [row for row in confidence if row['precision'] == label]
        x = [phases.index(row['phase']) + shift for row in data]
        y = [100 * float(row['mean']) for row in data]
        intervals = [[v - 100 * float(row['ci95_low']) for v, row in zip(y, data)],
                     [100 * float(row['ci95_high']) - v for v, row in zip(y, data)]]
        axes[0].errorbar(x, y, yerr=intervals, fmt='o', capsize=4, color=color, label=label.upper())
        data = [row for row in traces if row['precision'] == label
                and row['probe_serial_context_rate_ratio'] != 'UNAVAILABLE']
        axes[1].bar([trace_phases.index(row['phase_condition']) + shift for row in data],
                    [100 * float(row['probe_serial_context_rate_ratio']) for row in data],
                    width=.29, color=color)
    axes[0].axhline(0, color='#848484', linewidth=.8)
    axes[0].set_xticks(range(len(phases)), [phase.capitalize() for phase in phases])
    axes[0].set_ylabel('Complete request wall time change (%)')
    axes[0].set_title('P0 paired blocks, 95% t interval')
    axes[1].set_xticks(range(len(trace_phases)), [phase.capitalize() for phase in trace_phases])
    axes[1].set_title('P2 same-stage serial control')
    axes[1].set_ylabel('Probe kernel throughput retained (%)')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=2, frameon=False)
    fig.suptitle('Complete E2 inference with a fixed 64-CTA BF16 Tensor probe')
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)


if __name__ == '__main__':
    main()
