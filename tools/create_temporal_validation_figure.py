"""Render monthly accuracy and fitting variation from aggregate temporal results."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run():
    source = ROOT / 'results/company2_temporal/fold_metrics_aggregated.csv'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metrics = pd.read_csv(source)
    months = ['2026-04', '2026-05', '2026-07', '2026-08']
    labels = ['April\n(n=5)', 'May\n(n=10)', 'July\n(n=16)', 'August\n(n=17)']
    methods = [('ridge', 'Ridge', '#009E73', 'o'),
               ('ridge_projection', 'Ridge + Projection', '#007450', 's'),
               ('data_driven_projection', 'Data-driven NN + Projection', '#0072B2', 'D'),
               ('domain_informed', 'Domain-informed NN', '#D55E00', '^')]
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout='constrained')
    for axis, completeness in zip(axes, [1., .1]):
        for offset, (method, label, color, marker) in zip(np.linspace(-.18, .18, 4), methods):
            rows = metrics[(metrics.target == 'elapsed_hours') &
                           (metrics.completeness == completeness) & (metrics.method == method)]
            if rows.fold.duplicated().any() or set(rows.fold) != set(months):
                raise ValueError('Expected one aggregate per completed month and method')
            rows = rows.set_index('fold').loc[months]
            axis.errorbar(np.arange(4) + offset, rows.MAE, yerr=rows.MAE_seed_sd,
                          fmt=marker, color=color, capsize=2.5, markersize=4, label=label)
        axis.set(title=f'{completeness:.0%} training and validation labels',
                 ylabel='Elapsed-duration MAE (h)', xlabel='Test month in 2026', ylim=(0, None))
        axis.set_xticks(np.arange(4), labels)
        axis.grid(axis='y', alpha=.25)
    axes[0].legend(fontsize=6.5)
    out = ROOT / 'figures'
    for extension in ['pdf', 'png']:
        figure.savefig(out / f'temporal_validation.{extension}', dpi=200)
    plt.close(figure)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    provenance = {'source': str(source.relative_to(ROOT)), 'source_sha256': digest,
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'error_bars': 'Sample standard deviation over five paired fitting/masking seeds; not confidence intervals',
                  'test_months': months, 'unique_test_operations': 48}
    (out / 'temporal_validation.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    run()
