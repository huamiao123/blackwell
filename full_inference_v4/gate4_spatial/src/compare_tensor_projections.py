"""Compare the fixed 64-CTA BF16 Tensor probe across verified projections."""

import argparse
from pathlib import Path

from g4_common import V4, read_csv, write_csv


METRICS = ('primary_interference', 'probe_throughput_retention', 'joint_speedup')


def compare(root, operators):
    inventory = read_csv(V4 / 'gate4_residency/analysis/projection_inventory.csv')
    rows = []
    for operator in operators:
        label = operator.removesuffix('_proj')
        p0_name = 'qkv_grid64_fixed' if label == 'qkv' else f'{label}_grid64'
        p0_path = root / 'analysis' / f'tensor_light_{p0_name}_p0_statistics.csv'
        p2_name = 'grid64_tensor_light_trace' if label == 'qkv' else f'{operator}_grid64_tensor_light_trace'
        p2_path = root / 'analysis' / p2_name / 'qkv_pilot_summary.csv'
        estimates = read_csv(p0_path)
        p2 = read_csv(p2_path)
        intervals = {(r['case'], r['mode'], r['role']): r
                     for r in read_csv(p2_path.with_name('qkv_pilot_intervals.csv'))}
        for precision in ('nvfp4', 'bf16'):
            primary, = (r for r in inventory if
                        (r['precision'], r['logical_op'], int(r['layer_id'])) == (precision, operator, 18))
            for order in ('ab', 'ba'):
                activity, = (r for r in p2 if r['case'] == f'{precision}_{operator}'
                             and r['mode'] == f'parallel_{order}')
                if int(activity['probe_grid']) != 64:
                    raise ValueError('unexpected P2 probe grid')
                tail = int(activity['b_after_a_ns'])
                head = int(activity['b_before_a_ns'])
                a, b = (intervals[activity['case'], activity['mode'], role] for role in ('A', 'B'))
                row = dict(operator=operator, precision=precision, order=order,
                           primary_grid_ctas=int(primary['grid_ctas']), probe_grid_ctas=64,
                           p2_a_alone_ns=int(activity['a_alone_ns']), p2_b_alone_ns=int(activity['b_alone_ns']),
                           p2_overlap_ns=int(activity['overlap_ns']), p2_b_head_ns=head, p2_b_tail_ns=tail,
                           p2_b_start_minus_a_start_ns=int(b['start_ns']) - int(a['start_ns']),
                           p2_b_finish_slack_ns=int(a['end_ns']) - int(b['end_ns']),
                           p2_b_completed_inside_a=(head == 0 and tail == 0),
                           p0_source=p0_path.relative_to(root).as_posix(),
                           p2_source=p2_path.relative_to(root).as_posix())
                for metric in METRICS:
                    effect, = (r for r in estimates if (r['metric'], r['group'], r['order'], int(r['probe_grid']))
                               == (metric, precision, order, 64))
                    if int(effect['n_paired_blocks']) != 6:
                        raise ValueError('incomplete unprofiled paired blocks')
                    for field in ('mean', 'ci95_low', 'ci95_high'):
                        row[f'p0_{metric}_{field}'] = float(effect[field])
                rows.append(row)
    return rows


def plot(rows, output):
    import matplotlib.pyplot as plt

    operators = list(dict.fromkeys(row['operator'] for row in rows))
    labels = {'qkv_proj': 'QKV', 'o_proj': 'O projection', 'down_proj': 'FFN Down', 'gate_up_proj': 'Gate/Up'}
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.0), sharex=True, layout='constrained')
    settings = [('nvfp4', 'ab', -0.24, '#ad551e', 'o'), ('nvfp4', 'ba', -0.08, '#ad551e', 's'),
                ('bf16', 'ab', 0.08, '#295689', 'o'), ('bf16', 'ba', 0.24, '#295689', 's')]
    for ax, metric, label in zip(axes, METRICS[:2], ('Main projection change (%)', 'Probe throughput retained (%)')):
        for precision, order, shift, color, marker in settings:
            data = [row for row in rows if (row['precision'], row['order']) == (precision, order)]
            x = [operators.index(row['operator']) + shift for row in data]
            y = [100 * row[f'p0_{metric}_mean'] for row in data]
            error = [[value - 100 * row[f'p0_{metric}_ci95_low'] for value, row in zip(y, data)],
                     [100 * row[f'p0_{metric}_ci95_high'] - value for value, row in zip(y, data)]]
            ax.errorbar(x, y, yerr=error, color=color, marker=marker, capsize=3, linestyle='none',
                        label=f'{precision.upper()} {order.upper()}')
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=0.2)
    axes[0].axhline(5, linestyle='--', linewidth=0.9, color='#6b747c', label='5% loss budget')
    axes[1].axhline(100, linestyle='--', linewidth=0.9, color='#6b747c')
    axes[1].set_xticks(range(len(operators)), [labels[op] for op in operators])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=5, frameon=False, fontsize=8)
    fig.suptitle('Identical 64-CTA BF16 Tensor probe across projections\n'
                 'Unprofiled paired blocks, means and 95% t intervals')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--operators', nargs='+', choices=('qkv_proj', 'o_proj', 'down_proj', 'gate_up_proj'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figure', type=Path)
    args = p.parse_args()
    rows = compare(args.root, args.operators)
    write_csv(args.output, rows)
    if args.figure:
        plot(rows, args.figure)
    print(f'Joined {len(rows)} independently scoped P0/P2 projection comparisons')


if __name__ == '__main__':
    main()
