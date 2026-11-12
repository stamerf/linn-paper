"""Aggregation and observation availability tests for the industrial casting case."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from linn.data.company2_case import Company2CaseEncoder, prepare_casting_case
from linn.data.loaders import load_company2


def config():
    return json.loads(Path("configs/workstation.json").read_text())


def tables():
    confirmations, resources = [], []
    for number, day in [("1", "2026-02-01"), ("2", "2026-06-10"), ("3", "2026-08-01")]:
        # A quantity-only record plus two processing records on the same operation.
        for index, quantity, start, end, hours in [
            ("q", "100", day+"T12:00", day+"T12:00", "0"),
            ("a", "0", day+"T08:00", day+"T09:00", "1"),
            ("b", "0", day+"T10:00", day+"T12:00", "2")]:
            row = dict(afo_lfd_nr=number+index, afo_art_nr="P1", afo_tr="0",
                       afo_teg=hours, afo_tg=hours, afo_bab_typ="20", afo_bab_nr=number,
                       afo_bab_afo_nr="10", bab_belegart=None, afo_beginn=start,
                       afo_ende=end, afo_datum=end, afo_menge=quantity, afo_ausschuss="0")
            confirmations.append(row)
            resources.append(dict(afo_lfd_nr=number+index, res_lfd_nr=number+index,
                kst_typ="10", kst_roeders_kstgrp="APS", res_kst_id="PSK 01",
                res_beginn=start, res_ende=end, afo_datum=end, res_zeit=hours,
                res_zeitanteil="100", afo_bab_typ="20", afo_bab_nr=number,
                afo_bab_afo_nr="10", kst_bez1="Press"))
    return {"afoRows": pd.DataFrame(confirmations), "resourceRows": pd.DataFrame(resources)}


def test_aggregate_unique_quantity_and_cumulative_elapsed_not_span():
    source = tables()
    source["afoRows"].loc[source["afoRows"].afo_lfd_nr.eq("1q"), "afo_ausschuss"] = "5"
    source["afoRows"] = pd.concat([source["afoRows"], source["afoRows"].iloc[:1]], ignore_index=True)
    trim = source["resourceRows"].copy()
    trim["res_lfd_nr"] += "trim"
    trim["res_kst_id"] = "PF 01"
    trim["kst_roeders_kstgrp"] = "APM"
    source["resourceRows"] = pd.concat([source["resourceRows"], trim], ignore_index=True)
    result, audit = prepare_casting_case(source, config())
    assert len(result) == 3
    assert result.confirmed_quantity.tolist() == [105, 100, 100]
    assert result.confirmed_good_quantity.tolist() == [100, 100, 100]
    assert result.confirmed_scrap_quantity.tolist() == [5, 0, 0]
    assert result.elapsed_hours.tolist() == [3, 3, 3]
    assert result.booked_hours.tolist() == [3, 3, 3]
    assert ((result.end-result.start).dt.total_seconds()/3600).tolist() == [4, 4, 4]
    assert result.n_bookings.tolist() == [2, 2, 2]
    assert result.split.tolist() == ["train", "validation", "test"]
    assert audit.attrs["exclusion_counts"]["eligible_operations"] == 3


@pytest.mark.parametrize("field,value,flag", [
    ("afo_menge", "-10", "quantity_negative"),
    ("afo_ausschuss", "-1", "quantity_negative"),
    ("afo_menge", None, "quantity_missing"),
    ("afo_ausschuss", "inf", "quantity_missing"),
    ("afo_datum", None, "quantity_posting_missing"),
    ("afo_art_nr", "P2", "ambiguous_product"),
])
def test_quantity_corrections_missingness_and_product_conflicts_quarantine_group(field, value, flag):
    source = tables()
    source["afoRows"].loc[source["afoRows"].afo_lfd_nr.eq("1q"), field] = value
    result, audit = prepare_casting_case(source, config())
    assert "20:1" not in result.order_id.tolist()
    assert bool(audit.loc[audit.order_id.eq("20:1"), flag].iloc[0])


@pytest.mark.parametrize("field,value,flag", [
    ("res_ende", None, "invalid_processing_booking"),
    ("res_zeit", "-1", "invalid_processing_booking"),
    ("res_kst_id", "PSK 02", "ambiguous_press"),
])
def test_one_invalid_booking_or_ambiguous_press_rejects_whole_operation(field, value, flag):
    source = tables()
    source["resourceRows"].loc[source["resourceRows"].res_lfd_nr.eq("1a"), field] = value
    result, audit = prepare_casting_case(source, config())
    assert "20:1" not in result.order_id.tolist()
    assert bool(audit.loc[audit.order_id.eq("20:1"), flag].iloc[0])


def test_mixed_setup_processing_quarantines_group_but_pure_setup_does_not():
    for hours, should_exclude in [("1", True), ("0", False)]:
        source = tables()
        row = source["afoRows"].iloc[0].copy()
        row["afo_lfd_nr"], row["afo_menge"], row["afo_tr"], row["afo_teg"] = "1s", "0", "1", hours
        resource = source["resourceRows"].iloc[0].copy()
        resource["afo_lfd_nr"], resource["res_lfd_nr"] = "1s", "1s"
        source["afoRows"] = pd.concat([source["afoRows"], pd.DataFrame([row])], ignore_index=True)
        source["resourceRows"] = pd.concat([source["resourceRows"], pd.DataFrame([resource])], ignore_index=True)
        result, audit = prepare_casting_case(source, config())
        assert ("20:1" not in result.order_id.tolist()) == should_exclude


def test_quantity_posting_and_other_operation_extend_global_split_bounds():
    source = tables()
    source["afoRows"].loc[source["afoRows"].afo_lfd_nr.eq("1q"), "afo_datum"] = "2026-06-02"
    later = source["afoRows"].iloc[3].copy()
    later["afo_lfd_nr"], later["afo_bab_afo_nr"], later["afo_datum"] = "2later", "20", "2026-07-20"
    later["afo_beginn"], later["afo_ende"] = None, None
    source["afoRows"] = pd.concat([source["afoRows"], pd.DataFrame([later])], ignore_index=True)
    result, _ = prepare_casting_case(source, config())
    assert result.split.tolist() == ["purged", "purged", "test"]


def test_preperiod_history_quarantines_complete_order():
    source = tables()
    source["afoRows"].loc[source["afoRows"].afo_lfd_nr.eq("1q"), "afo_beginn"] = "2025-12-30"
    result, audit = prepare_casting_case(source, config())
    assert "20:1" not in result.order_id.tolist()
    assert bool(audit.loc[audit.order_id.eq("20:1"), "outside_complete_period"].iloc[0])


def test_case_encoder_training_fit_and_feature_allowlist():
    frame, _ = prepare_casting_case(tables(), config())
    train = frame[frame.split.eq("train")]
    encoder = Company2CaseEncoder(log_quantity=True).fit(train)
    changed = frame.copy()
    for column in ["elapsed_hours", "booked_hours", "n_bookings", "n_confirmations"]:
        changed[column] = 99999
    changed["end"] = pd.Timestamp("2099-01-01")
    np.testing.assert_array_equal(encoder.transform(frame), encoder.transform(changed))
    assert encoder.metadata()["fitted_record_ids"] == train.record_id.tolist()
    assert encoder.metadata()["numeric_mean"] == pytest.approx([np.log1p(100)])
    changed["machine"] = "UNKNOWN"
    assert np.isfinite(encoder.transform(changed)).all()
    with pytest.raises(ValueError, match="training"):
        Company2CaseEncoder().fit(frame)
    ablated = Company2CaseEncoder(include_quantity=False).fit(train)
    changed = frame.copy(); changed["confirmed_quantity"] = np.nan
    np.testing.assert_array_equal(ablated.transform(frame), ablated.transform(changed))


@pytest.mark.skipif(not Path("daten/manifest.json").exists(), reason="Private Company 2 export absent")
def test_real_case_cardinality_temporal_purge_and_finite_features():
    raw, _, _ = load_company2("daten")
    frame, audit = prepare_casting_case(raw, config())
    assert frame.record_id.is_unique
    assert frame.groupby("order_id").split.nunique().max() == 1
    assert frame.n_presses.eq(1).all()
    assert np.isfinite(frame[["elapsed_hours", "booked_hours", "confirmed_quantity"]]).all().all()
    assert frame[["elapsed_hours", "booked_hours", "confirmed_quantity"]].gt(0).all().all()
    train = frame[frame.split.eq("train")]
    validation = frame[frame.split.eq("validation")]
    test = frame[frame.split.eq("test")]
    assert train.global_order_last.lt("2026-06-01").all()
    assert validation.global_order_first.ge("2026-06-01").all()
    assert validation.global_order_last.lt("2026-07-16").all()
    assert test.global_order_first.ge("2026-07-16").all()
    encoder = Company2CaseEncoder().fit(train)
    assert np.isfinite(encoder.transform(frame)).all()
    assert audit.attrs["exclusion_counts"]["eligible_operations"] == len(frame)
