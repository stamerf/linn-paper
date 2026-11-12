import pandas as pd
import pytest

from linn.experiments.company2_temporal import monthly_split
from linn.experiments.company2_case import select_case


def test_monthly_split_uses_complete_global_orders_and_both_test_boundaries():
    first = ['2026-01-01', '2026-03-01', '2026-04-01', '2026-02-28',
             '2026-03-31', '2026-04-30', '2026-05-01', '2025-12-31', '2026-03-31']
    last = ['2026-02-28', '2026-03-31', '2026-04-30', '2026-03-01',
            '2026-04-01', '2026-05-01', '2026-05-10', '2026-01-02', '2026-04-01']
    frame = pd.DataFrame({'record_id': list('abcdefghi'),
                          'order_id': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'e'],
                          'global_order_first': pd.to_datetime(first),
                          'global_order_last': pd.to_datetime(last),
                          'start': pd.to_datetime(['2026-04-10']*9)})
    split, bounds = monthly_split(frame, '2026-04-01')
    assert split.split.tolist() == ['train', 'validation', 'test'] + ['purged']*6
    assert bounds['validation_start'] == '2026-03-01'
    assert bounds['test_end_exclusive'] == '2026-05-01'
    assert split.groupby('order_id').split.nunique().max() == 1
    # Retained operation timestamps must not override all-raw-record order bounds.
    frame['start'] = pd.Timestamp('2026-01-01')
    same, _ = monthly_split(frame, '2026-04-01')
    assert same.split.tolist() == split.split.tolist()


def test_monthly_selection_cannot_be_changed_by_test_volume_or_target_values():
    frame = pd.DataFrame({'machine': ['A']*20 + ['B']*20 + ['C']*100,
                          'split': ['train']*15 + ['validation']*5 + ['train']*14 +
                                   ['validation']*6 + ['test']*100})
    selected, machines, _ = select_case(frame, {'min_train': 15, 'min_validation': 5})
    assert machines == ['A']
    _, past_only, _ = select_case(frame[frame.split.ne('test')], {'min_train': 15, 'min_validation': 5})
    assert machines == past_only
    assert len(selected) == 20


def test_monthly_split_rejects_inconsistent_order_bounds():
    frame = pd.DataFrame({'record_id': ['a', 'b'], 'order_id': ['same', 'same'],
                          'global_order_first': pd.to_datetime(['2026-01-01', '2026-02-01']),
                          'global_order_last': pd.to_datetime(['2026-03-01', '2026-03-01'])})
    with pytest.raises(ValueError, match='Inconsistent global order bounds'):
        monthly_split(frame, '2026-04-01')
