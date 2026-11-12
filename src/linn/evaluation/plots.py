from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def exploratory_plots(frame, metadata, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    machines = sorted(frame.machine.unique())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    counts = frame.groupby("machine").agg(bookings=("record_id", "size"), orders=("order_id", "nunique")).reindex(machines)
    x = np.arange(len(machines))
    axes[0,0].bar(x-.18, counts.bookings, .36, label="Processing bookings")
    axes[0,0].bar(x+.18, counts.orders, .36, label="Production orders")
    axes[0,0].set(xticks=x, xticklabels=machines, ylabel="Count", title="Retained cohort (Jan–Aug 2026)")
    axes[0,0].legend()
    for machine, color in zip(machines, colors):
        g = frame[frame.machine.eq(machine)]
        axes[0,1].hist(g.elapsed_hours, bins=np.linspace(0,24,49), histtype="step", density=True, label=machine, color=color)
        weekly = g.set_index("start").record_id.resample("W-MON").count()
        axes[1,0].plot(weekly.index.to_numpy(), weekly.values, label=machine, color=color)
    axes[0,1].set(xlabel="Elapsed processing-booking hours", ylabel="Density", title="No part-cycle interpretation")
    axes[0,1].legend()
    axes[1,0].set(ylabel="Bookings per week", title="Temporal coverage")
    axes[1,0].tick_params(axis="x", rotation=30)
    quality = metadata.set_index("machine").loc[machines]
    for offset, col, label in [(-.25,"missing_fraction","Missing timestamp"), (0,"overlap_fraction","Overlap (>1 s)"),
                                (.25,"accounting_violation_fraction","Accounting violation")]:
        axes[1,1].bar(x+offset, quality[col]*100, .25, label=label)
    axes[1,1].set(xticks=x, xticklabels=machines, ylabel="Bookings (%)", title="Quality audit (full cohort)")
    axes[1,1].legend(fontsize=8)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"workstation_overview.{extension}", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(machines), figsize=(12,4), constrained_layout=True)
    for ax, machine, color in zip(np.atleast_1d(axes), machines, colors):
        g = frame[frame.machine.eq(machine)]
        ax.scatter(g.elapsed_hours, g.booked_hours, s=10, alpha=.45, color=color)
        limit = max(g.elapsed_hours.max(), g.booked_hours.max())*1.05
        ax.plot([0,limit], [0,limit], "k--", linewidth=1)
        ax.set(title=machine, xlabel="Elapsed hours", ylabel="Booked hours", xlim=(0,limit), ylim=(0,limit))
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"booking_accounting.{extension}", dpi=200)
    plt.close(fig)
