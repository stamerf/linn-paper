import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_company2_projection import analyze


def test_saved_industrial_comparison_aligns_cohort_and_preserves_private_inputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ['tools/evaluate_company2_projection.py', 'src/linn/evaluation/projection.py',
                 'src/linn/evaluation/metrics.py']:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / name).read_bytes())
    config = tmp_path / 'configs/company2_case.json'
    config.parent.mkdir()
    config.write_text(json.dumps({'completeness': [1., .1], 'evaluation_seeds': [101, 202],
                                  'rounding_tolerance_hours': .02}))
    source = tmp_path / 'results/company2_case'
    (source / 'runs').mkdir(parents=True)
    for name, value in [('selected_configs.json', {}), ('protocol.json', {}),
                        ('environment.json', {}), ('raw_inventory.json', {}), ('cohort_summary.json', {}),
                        ('validation.json', {'protected_unchanged': True, 'primary_case_exploratory': True})]:
        (source / name).write_text(json.dumps(value))
    for level in [1., .1]:
        for seed in [101, 202]:
            for method in ['quantity_linear', 'data_driven', 'domain_informed']:
                frame = pd.DataFrame({'record_id': ['secret_a', 'secret_b', 'secret_c', 'secret_d'],
                                      'order_id': ['private_1', 'private_2', 'private_3', 'private_4'],
                                      'machine': ['M1', 'M1', 'M2', 'M2'], 'split': ['test']*4,
                                      'elapsed_observed': [1., 3., 2., 4.],
                                      'booked_observed': [.8, 2.9, 1.8, 3.9],
                                      'elapsed_predicted': [-1., 2., 2., 4.],
                                      'booked_predicted': [1., 4., 1.8, 3.9]})
                frame.to_csv(source / 'runs' / f'{int(level*100):03d}_{seed}_q1_{method}_predictions.csv', index=False)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    analyze(tmp_path)
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    out = tmp_path / 'results/company2_projection'
    metrics = pd.read_csv(out / 'metrics.csv')
    corrected = metrics[metrics.method.str.endswith(('_projection', '_clipping'))]
    assert corrected.constraint_violation.eq(0).all()
    original = metrics[metrics.method.eq('data_driven') & metrics.machine.eq('pooled')]
    np.testing.assert_allclose(original.constraint_violation, .5)
    assert set(metrics.completeness) == {1., .1}
    assert set(metrics.seed) == {101, 202}
    for path in out.iterdir():
        # Aggregate public artifacts must not disclose individual operation IDs.
        text = path.read_text()
        assert 'secret_a' not in text
        assert 'private_1' not in text
