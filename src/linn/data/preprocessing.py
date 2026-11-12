"""Explicit booking cohort, audit flags, development-only selection and purged splits."""
import numpy as np
import pandas as pd


def parse_wall_time(series):
    # Mixed ISO precision is present in the source. Never infer one format from row 1.
    return pd.to_datetime(series, format="ISO8601", errors="coerce")


def unique_records(frame, key):
    clean = frame.drop_duplicates().copy()
    if clean[key].isna().any() or clean[key].duplicated().any():
        raise ValueError(f"Missing or conflicting {key}; refusing an ambiguous join")
    return clean


def preprocess(tables, config):
    afo = unique_records(tables["afoRows"], "afo_lfd_nr")
    res = unique_records(tables["resourceRows"], "res_lfd_nr")
    concrete = (res.kst_typ.eq("10") & res.kst_roeders_kstgrp.notna()
                & res.res_kst_id.str.contains(r"\d", na=False))
    columns = ["afo_lfd_nr", "afo_art_nr", "afo_tr", "afo_teg", "afo_tg",
               "afo_bab_typ", "afo_bab_nr", "afo_bab_afo_nr", "bab_belegart"]
    frame = res.loc[concrete].merge(afo[columns], on="afo_lfd_nr", how="left",
                                  suffixes=("", "_confirmation"), validate="many_to_one", indicator=True)
    frame["start"] = parse_wall_time(frame.res_beginn)
    frame["end"] = parse_wall_time(frame.res_ende)
    frame["posting"] = parse_wall_time(frame.afo_datum)
    for col in ["afo_tr", "afo_teg", "afo_tg", "res_zeit", "res_zeitanteil"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["elapsed_hours"] = (frame.end - frame.start).dt.total_seconds() / 3600
    frame["booked_hours"] = frame.res_zeit
    frame["machine"] = frame.res_kst_id
    frame["record_id"] = frame.res_lfd_nr
    frame["order_id"] = frame.afo_bab_typ + ":" + frame.afo_bab_nr
    frame["product"] = frame.afo_art_nr
    frame["operation"] = frame.afo_bab_afo_nr
    # The fallback is ONLY for quality denominators, never for reconstructing a timestamp.
    anchor = frame.start.fillna(frame.posting)
    frame["in_period"] = anchor.ge(pd.Timestamp(config["period_start"])) & anchor.lt(pd.Timestamp(config["period_end_exclusive"]))
    frame["development"] = anchor.lt(pd.Timestamp(config["validation_start"]))
    frame["processing"] = (frame["product"].notna() & ~frame["product"].str.startswith("#", na=False)
                           & frame.afo_tr.eq(0) & frame.afo_teg.gt(0))
    frame["missing_timestamp"] = frame.start.isna() | frame.end.isna()
    frame["invalid_order"] = frame.elapsed_hours.lt(0)
    frame["zero_interval"] = frame.elapsed_hours.eq(0)
    frame["extreme_interval"] = frame.elapsed_hours.gt(config["max_elapsed_hours"])
    frame["invalid_booked"] = ~np.isfinite(frame.booked_hours) | frame.booked_hours.le(0)
    frame["allocation_unsupported"] = ~frame.res_zeitanteil.between(0, 100)
    frame["join_inconsistent"] = frame._merge.ne("both")
    for col in ["afo_bab_typ", "afo_bab_nr", "afo_bab_afo_nr"]:
        frame["join_inconsistent"] |= frame[col].ne(frame[col + "_confirmation"])
    frame["accounting_violation"] = frame.booked_hours.gt(frame.elapsed_hours + config["rounding_tolerance_hours"])
    # Do not delete records merely because they violate the proposed soft constraint.
    flags = ["missing_timestamp", "invalid_order", "zero_interval", "extreme_interval",
             "invalid_booked", "allocation_unsupported", "join_inconsistent"]
    frame["valid"] = frame.in_period & frame.processing & ~frame[flags].any(axis=1)
    frame["exclusion_reason"] = frame[flags].apply(lambda row: ";".join(row.index[row]), axis=1)
    clean = frame[frame.valid].copy().sort_values(["machine", "start", "record_id"]).reset_index(drop=True)
    if clean.empty:
        raise ValueError("No valid processing intervals")
    return clean, frame


def machine_metadata(clean, audit, development_only=True):
    rows = []
    scope = audit[audit.in_period & audit.processing]
    if development_only:
        scope = scope[scope.development]
        clean = clean[clean.development]
    for machine, raw in scope.groupby("machine", sort=True):
        g = clean[clean.machine.eq(machine)].sort_values(["start", "record_id"])
        if g.empty:
            continue
        overlaps = g.start.lt(g.end.cummax().shift() - pd.Timedelta(seconds=1))
        rows.append(dict(machine=machine, description=raw.kst_bez1.iloc[0], group=raw.kst_roeders_kstgrp.iloc[0],
                         n_raw=len(raw), n_records=len(g), n_orders=g.order_id.nunique(), n_products=g["product"].nunique(),
                         valid_fraction=len(g)/len(raw), missing_fraction=raw.missing_timestamp.mean(),
                         overlap_fraction=overlaps.mean(), accounting_violation_fraction=g.accounting_violation.mean(),
                         elapsed_median=g.elapsed_hours.median(), elapsed_std=g.elapsed_hours.std(),
                         booked_median=g.booked_hours.median(), first_start=str(g.start.min()), last_end=str(g.end.max())))
    return pd.DataFrame(rows)


def select_machines(metadata, settings):
    m = metadata.copy()
    m["eligible"] = (m.group.eq(settings["group"]) & m.n_records.ge(settings["min_records"])
                     & m.n_orders.ge(settings["min_orders"]) & m.n_products.ge(settings["min_products"])
                     & m.valid_fraction.ge(settings["min_valid_fraction"]) & m.elapsed_std.gt(0))
    m["score"] = np.log1p(m.n_records) * m.valid_fraction * (1-m.overlap_fraction)
    ranked = m[m.eligible].sort_values(["score", "machine"], ascending=[False, True])
    # Distinct press descriptions introduce nominal capacity diversity without hand-picking IDs.
    selected = ranked.drop_duplicates("description").head(settings["count"]).machine.tolist()
    if len(selected) != settings["count"]:
        raise ValueError("Insufficient eligible distinct workstations; review selection thresholds")
    m["selected"] = m.machine.isin(selected)
    return selected, m


def chronological_split(frame, validation_start, test_start):
    """Global boundaries; purge whole orders spanning either boundary (across machines)."""
    v, t = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    if v >= t:
        raise ValueError("Validation boundary must precede test boundary")
    if frame[["start", "end", "order_id"]].isna().any().any():
        raise ValueError("Splits require complete timestamps and order identifiers")
    bounds = frame.groupby("order_id").agg(first=("start", "min"), last=("end", "max"))
    bounds["split"] = "purged"
    bounds.loc[bounds["last"].lt(v), "split"] = "train"
    bounds.loc[bounds["first"].ge(v) & bounds["last"].lt(t), "split"] = "validation"
    bounds.loc[bounds["first"].ge(t), "split"] = "test"
    out = frame.copy()
    out["split"] = out.order_id.map(bounds["split"])
    return out
