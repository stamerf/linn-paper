import numpy as np
import pandas as pd
import pytest

from tools.evaluate_confirmation_projection import METHODS, summarize_paired


def test_projection_comparison_resamples_worlds_and_averages_paired_effects():
    rows = []
    for world in range(20):
        for i, machine in enumerate(['A', 'B', 'C']):
            for method in METHODS:
                error = 2.**i
                if method == 'domain_informed':
                    error *= 1 - .1*(i+1)
                rows.append({'world': world, 'machine': machine, 'completeness': .05,
                             'target': 'elapsed_hours', 'method': method, 'MAE': error})
    metrics = pd.DataFrame(rows)
    machine_effects, worlds, intervals = summarize_paired(metrics, repetitions=1000)
    selected = intervals[intervals.candidate.eq('domain_informed') &
                         intervals.baseline.eq('data_driven_projection') &
                         intervals.metric.eq('relative_improvement')].iloc[0]
    assert selected['n_worlds'] == 20
    assert selected['mean'] == pytest.approx(.2)
    assert selected['ci_low'] == pytest.approx(.2)
    assert selected['ci_high'] == pytest.approx(.2)
    assert selected['positive_worlds'] == 20
    assert len(worlds[worlds.candidate.eq('domain_informed') &
                      worlds.baseline.eq('data_driven_projection')]) == 20
    assert len(machine_effects[machine_effects.candidate.eq('domain_informed') &
                              machine_effects.baseline.eq('data_driven_projection')]) == 60
    pooled = metrics.groupby('method').MAE.mean()
    assert not np.isclose(selected['mean'], 1-pooled.domain_informed/pooled.data_driven_projection)
    with pytest.raises(ValueError, match='Duplicate'):
        summarize_paired(pd.concat([metrics, metrics.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match='finite positive'):
        summarize_paired(metrics.drop(metrics.index[0]))
