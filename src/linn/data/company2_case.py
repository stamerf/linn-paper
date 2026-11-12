"""Conservative retrospective casting-operation case from Company 2.

An example is one production-order/operation on one APS press. Confirmed good
quantity and scrap are summed once at the unique AFO-confirmation grain, before
joining resource bookings. They are retrospective observations, not planned
workload. The targets are sums of valid processing-booking elapsed and booked
hours; cumulative elapsed hours are NOT the wall-clock span of the operation.

The existing booking adapter and neural/training core are deliberately unchanged.
"""
import inspect

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocessing import parse_wall_time, preprocess, unique_records


KEY = ["order_id", "operation"]


def _identifiers(frame):
    frame = frame.copy()
    frame["order_id"] = frame.afo_bab_typ + ":" + frame.afo_bab_nr
    frame["operation"] = frame.afo_bab_afo_nr
    if frame[KEY].isna().any().any():
        raise ValueError("Missing production-order or operation identifier")
    return frame


def prepare_casting_case(tables, config):
    """Return (prepared examples, operation audit), including conservative splits.

    Every operation containing a negative good/scrap correction, missing numeric
    quantity, ambiguous product/press, or invalid processing booking is excluded.
    Press uniqueness is checked within APS, since presses and trimming machines
    legitimately share operation keys. Setup/mixed bookings are outside the target.

    All productive raw AFO start/end/posting dates and concrete-resource endpoints
    define GLOBAL order bounds, even for rows excluded from the learning cohort.
    This prevents future quantity postings or other-machine bookings from crossing
    split boundaries unnoticed. Missing quantity-event posting dates and impossible
    group period bounds are quarantined. Do not re-split the returned frame with a
    function using only its retained examples. No empirical outcome cutoffs are fit.
    """
    afo = _identifiers(unique_records(tables["afoRows"], "afo_lfd_nr"))
    resources = unique_records(tables["resourceRows"], "res_lfd_nr")
    productive = afo.afo_art_nr.notna() & ~afo.afo_art_nr.str.startswith("#", na=False)
    afo = afo.loc[productive].copy()
    for name in ["afo_menge", "afo_ausschuss", "afo_tr", "afo_teg"]:
        afo[name] = pd.to_numeric(afo[name], errors="coerce")
    afo["quantity_negative"] = afo[["afo_menge", "afo_ausschuss"]].lt(0).any(axis=1)
    afo["quantity_missing"] = ~np.isfinite(afo[["afo_menge", "afo_ausschuss"]]).all(axis=1)
    afo["posting"] = parse_wall_time(afo.afo_datum)
    afo["quantity_event"] = afo[["afo_menge", "afo_ausschuss"]].ne(0).any(axis=1)
    afo["quantity_posting_missing"] = afo.quantity_event & afo.posting.isna()
    for name in ["afo_beginn", "afo_ende"]:
        afo[name] = parse_wall_time(afo[name])
    afo["observation_first"] = afo[["afo_beginn", "afo_ende", "posting"]].min(axis=1)
    afo["observation_last"] = afo[["afo_beginn", "afo_ende", "posting"]].max(axis=1)

    # This is the ONLY quantity aggregation. Resource multiplicity cannot inflate it.
    quantities = afo.groupby(KEY).agg(
        confirmed_good_quantity=("afo_menge", "sum"),
        confirmed_scrap_quantity=("afo_ausschuss", "sum"),
        quantity_negative=("quantity_negative", "any"),
        quantity_missing=("quantity_missing", "any"),
        quantity_posting_missing=("quantity_posting_missing", "any"),
        n_products=("afo_art_nr", "nunique"), product=("afo_art_nr", "first"),
        n_confirmations=("afo_lfd_nr", "size"))
    quantities["confirmed_quantity"] = (quantities.confirmed_good_quantity
                                         + quantities.confirmed_scrap_quantity)

    concrete = (resources.kst_typ.eq("10") & resources.kst_roeders_kstgrp.notna()
                & resources.res_kst_id.str.contains(r"\d", na=False))
    raw = resources.loc[concrete].merge(
        afo[["afo_lfd_nr", "order_id", "operation", "afo_tr", "afo_teg"]], on="afo_lfd_nr",
        how="inner", validate="many_to_one")
    raw["observation_first"] = pd.concat(
        [parse_wall_time(raw.res_beginn), parse_wall_time(raw.res_ende)], axis=1).min(axis=1)
    raw["observation_last"] = pd.concat(
        [parse_wall_time(raw.res_beginn), parse_wall_time(raw.res_ende)], axis=1).max(axis=1)
    global_bounds = pd.concat([
        afo[["order_id", "observation_first", "observation_last"]],
        raw[["order_id", "observation_first", "observation_last"]]
    ]).groupby("order_id").agg(
        global_order_first=("observation_first", "min"),
        global_order_last=("observation_last", "max"))
    assignment = raw[raw.kst_roeders_kstgrp.eq("APS")].groupby(KEY).agg(
        n_presses=("res_kst_id", "nunique"), machine=("res_kst_id", "first"))
    raw["mixed_setup_processing"] = raw.afo_tr.gt(0) & raw.afo_teg.gt(0)
    mixed = raw[raw.kst_roeders_kstgrp.eq("APS")].groupby(KEY).mixed_setup_processing.any()

    _, booking_audit = preprocess(tables, config)
    processing = booking_audit[
        booking_audit.kst_roeders_kstgrp.eq("APS") & booking_audit.processing].copy()
    operations = processing.groupby(KEY).agg(
        elapsed_hours=("elapsed_hours", "sum"), booked_hours=("booked_hours", "sum"),
        start=("start", "min"), end=("end", "max"), n_bookings=("record_id", "size"),
        all_bookings_valid=("valid", "all"))
    audit = operations.join(quantities).join(assignment).join(mixed).reset_index()
    audit = audit.merge(global_bounds, on="order_id", how="left", validate="many_to_one")
    audit["ambiguous_product"] = audit.n_products.ne(1)
    audit["ambiguous_press"] = audit.n_presses.ne(1)
    audit["invalid_processing_booking"] = ~audit.all_bookings_valid
    audit["nonpositive_quantity"] = audit.confirmed_quantity.le(0) | audit.confirmed_quantity.isna()
    begin, finish = pd.Timestamp(config["period_start"]), pd.Timestamp(config["period_end_exclusive"])
    audit["outside_complete_period"] = ~(audit.global_order_first.ge(begin)
                                         & audit.global_order_last.lt(finish))
    flags = ["ambiguous_product", "ambiguous_press", "invalid_processing_booking",
             "quantity_negative", "quantity_missing", "quantity_posting_missing",
             "nonpositive_quantity", "outside_complete_period", "mixed_setup_processing"]
    audit["eligible"] = ~audit[flags].fillna(True).any(axis=1)
    audit["exclusion_reason"] = audit[flags].fillna(True).apply(
        lambda row: ";".join(row.index[row]), axis=1)
    v, t = pd.Timestamp(config["validation_start"]), pd.Timestamp(config["test_start"])
    if not begin < v < t < finish:
        raise ValueError("Require period_start < validation_start < test_start < period_end")
    audit["split"] = "purged"
    audit.loc[audit.global_order_last.lt(v), "split"] = "train"
    audit.loc[audit.global_order_first.ge(v) & audit.global_order_last.lt(t), "split"] = "validation"
    audit.loc[audit.global_order_first.ge(t), "split"] = "test"
    audit["record_id"] = "APS:" + audit.order_id + ":" + audit.operation
    frame = audit[audit.eligible].copy().sort_values(["start", "record_id"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No eligible complete casting operations")
    audit.attrs["exclusion_counts"] = {name: int(audit[name].sum()) for name in flags}
    audit.attrs["exclusion_counts"].update(candidate_operations=len(audit),
                                          eligible_operations=len(frame),
                                          purged_eligible=int(frame.split.eq("purged").sum()))
    return frame, audit


class Company2CaseEncoder:
    """Train-only encoder with an explicit allowlist and optional quantity ablation.

    Quantity is an independent retained observation in this retrospective task.
    Counts, time sums/spans, target values, posting times, IDs and status are never
    features. Unknown validation/test category levels are encoded as all zeros.
    """
    def __init__(self, include_quantity=True, log_quantity=False, include_calendar=False):
        self.include_quantity = include_quantity
        self.log_quantity = log_quantity
        self.include_calendar = include_calendar

    def _features(self, frame):
        x = frame[["machine", "product", "operation"]].fillna("__missing__").astype(str).copy()
        if self.include_quantity:
            quantity = frame.confirmed_quantity.to_numpy(dtype=float)
            if not np.isfinite(quantity).all() or (quantity <= 0).any():
                raise ValueError("Confirmed quantity must be finite and positive")
            x["confirmed_quantity"] = np.log1p(quantity) if self.log_quantity else quantity
        if self.include_calendar:
            hour = frame.start.dt.hour + frame.start.dt.minute / 60
            x["hour_sin"] = np.sin(2*np.pi*hour/24)
            x["hour_cos"] = np.cos(2*np.pi*hour/24)
            x["weekday_sin"] = np.sin(2*np.pi*frame.start.dt.dayofweek/7)
            x["weekday_cos"] = np.cos(2*np.pi*frame.start.dt.dayofweek/7)
        return x

    def fit(self, train):
        if train.empty or not train.split.eq("train").all():
            raise ValueError("Fit requires a nonempty training split only")
        dense = "sparse_output" if "sparse_output" in inspect.signature(OneHotEncoder).parameters else "sparse"
        transforms = [("categories", OneHotEncoder(handle_unknown="ignore", **{dense: False}),
                       ["machine", "product", "operation"])]
        x = self._features(train)
        self.numeric_columns = [c for c in x if c not in ("machine", "product", "operation")]
        if self.numeric_columns:
            transforms.append(("numeric", StandardScaler(), self.numeric_columns))
        self.transformer = ColumnTransformer(transforms).fit(x)
        self.fitted_ids = train.record_id.tolist()
        return self

    def transform(self, frame):
        x = self.transformer.transform(self._features(frame)).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite encoded casting-case features")
        return x

    def metadata(self):
        result = {"fitted_record_ids": self.fitted_ids,
                  "feature_names": self.transformer.get_feature_names_out().tolist(),
                  "include_quantity": self.include_quantity, "log_quantity": self.log_quantity,
                  "include_calendar": self.include_calendar, "numeric_columns": self.numeric_columns}
        if self.numeric_columns:
            scaler = self.transformer.named_transformers_["numeric"]
            result.update(numeric_mean=scaler.mean_.tolist(), numeric_scale=scaler.scale_.tolist())
        return result
