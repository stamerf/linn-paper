"""Small synthetic single-server batch workstation; no Company 2 parameters are inferred.

The generator is outside the neural model and constraints. Truth/oracle columns
are audit-only: they must never enter features or constraint inputs.
"""
import numpy as np
import pandas as pd


def simulate_workstations(settings):
    n = int(settings["jobs_per_machine"])
    n_train = int(n*settings["train_fraction"])
    n_val = int(n*settings["validation_fraction"])
    if n_train < 1 or n_val < 1 or n_train+n_val >= n:
        raise ValueError("Simulation requires nonempty chronological splits")
    if settings["processing_log_noise_sigma"] < 0 or settings["arrival_mean_hours"] <= 0:
        raise ValueError("Invalid noise or arrival parameters")
    if len({m["machine"] for m in settings["machines"]}) != len(settings["machines"]):
        raise ValueError("Machine IDs must be unique")
    products = sorted(settings["product_work_factors"])
    if any(v <= 0 for v in settings["product_work_factors"].values()):
        raise ValueError("Product work factors must be positive")
    rows = []
    seeds = np.random.SeedSequence(settings["seed"]).spawn(len(settings["machines"]))
    origin = pd.Timestamp(settings["start_time"])
    for machine, seed in zip(settings["machines"], seeds):
        if machine["rate_units_per_hour"] <= 0:
            raise ValueError("Rate must be positive")
        rng = np.random.default_rng(seed)
        arrival, previous_end = origin, origin
        for i in range(n):
            arrival += pd.Timedelta(hours=float(rng.exponential(settings["arrival_mean_hours"])))
            start = max(arrival, previous_end)
            product = str(rng.choice(products))
            quantity = int(rng.integers(settings["minimum_quantity"],settings["maximum_quantity"]+1))
            hour = start.hour + start.minute/60
            calendar = 1+settings["calendar_amplitude"]*np.sin(2*np.pi*hour/24)
            expected_booked = (quantity/machine["rate_units_per_hour"] * settings["product_work_factors"][product]
                               * calendar * (1+settings["batch_nonlinearity"]*(quantity/100)**2))
            sigma = settings["processing_log_noise_sigma"]
            booked = expected_booked * float(rng.lognormal(-0.5*sigma*sigma, sigma))
            expected_pause = settings["pause_base_hours"]+settings["pause_per_unit_hours"]*quantity+settings["pause_random_mean_hours"]
            pause = (settings["pause_base_hours"]+settings["pause_per_unit_hours"]*quantity
                     +float(rng.exponential(settings["pause_random_mean_hours"])))
            elapsed = booked+pause
            end = start+pd.Timedelta(hours=elapsed)
            split = "train" if i<n_train else ("validation" if i<n_train+n_val else "test")
            identifier = f"{machine['machine']}:{i:05d}"
            rows.append({"record_id":identifier,"order_id":identifier,"machine":machine["machine"],
                         "product":product,"operation":"10","planned_quantity":quantity,
                         "arrival":arrival,"start":start,"end":end,"elapsed_hours":elapsed,"booked_hours":booked,
                         "split":split,"truth_queue_wait_hours":(start-arrival).total_seconds()/3600,
                         "truth_pause_hours":pause,"oracle_booked_hours":expected_booked,
                         "oracle_elapsed_hours":expected_booked+expected_pause})
            previous_end = end
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame[["elapsed_hours","booked_hours"]]).all().all():
        raise ValueError("Nonfinite simulation output")
    return frame
