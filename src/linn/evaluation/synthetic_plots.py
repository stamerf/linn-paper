from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {"median":("Median","#888888"), "limited_features":("NN without quantity","#CC79A7"),
           "data_driven":("NN with quantity","#0072B2"), "domain_informed":("NN + constraints","#D55E00"),
           "oracle_mean":("Known generator mean","#009E73")}


def plot_synthetic(frame, metrics, predictions, histories, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        for ext in ["png","pdf"]:fig.savefig(out/f"{name}.{ext}",dpi=180)
        plt.close(fig)
    m=metrics[(metrics.split=="test")&(metrics.target=="elapsed_hours")]
    machines=sorted(m.machine.unique());x=np.arange(len(machines))
    fig,axs=plt.subplots(1,3,figsize=(13,4.5))
    for i,(method,(label,color)) in enumerate(METHODS.items()):
        g=m[m.method==method].groupby("machine").agg(mae=("MAE","mean"),sd=("MAE","std"),r2=("R2","mean"),violation=("constraint_violation","mean")).reindex(machines)
        for ax,col in zip(axs,["mae","r2","violation"]):
            ax.bar(x+(i-2)*.16,g[col],.16,color=color,label=label,yerr=g.sd if col=="mae" else None)
    for ax in axs: ax.set_xticks(x);ax.set_xticklabels(machines)
    axs[0].set(ylabel="Elapsed MAE (h), mean ± seed SD")
    axs[1].set(ylabel="Elapsed R²");axs[2].set(ylabel="Any constraint violation fraction")
    fig.legend(*axs[0].get_legend_handles_labels(),loc="upper center",ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,0,1,.85))
    # Preserve legend room after layout.
    for ext in ["png","pdf"]:fig.savefig(out/f"synthetic_comparison.{ext}",dpi=180)
    plt.close(fig)
    seed=int(predictions.seed.min())
    for machine in machines:
        p=predictions[(predictions.machine==machine)&(predictions.seed==seed)&(predictions.split=="test")&(predictions.target=="elapsed_hours")]
        fig,axs=plt.subplots(1,3,figsize=(12,4))
        for method in ["limited_features","data_driven","domain_informed"]:
            g=p[p.method==method];label,color=METHODS[method]
            axs[0].scatter(g.observed,g.predicted,s=12,alpha=.45,label=label,color=color)
            axs[1].plot(np.arange(min(35,len(g))),g.predicted.iloc[:35].to_numpy(),label=label,color=color)
            h=histories[(histories.machine==machine)&(histories.seed==seed)&(histories.method==method)]
            axs[2].plot(h.epoch.to_numpy(),h.validation_mae_hours.to_numpy(),label=label,color=color)
        g=p[p.method=="data_driven"]
        lim=max(p.observed.max(),p.predicted.max())
        axs[0].plot([0,lim],[0,lim],"k--",lw=1)
        axs[1].plot(np.arange(min(35,len(g))),g.observed.iloc[:35].to_numpy(),"k--",label="Observed")
        axs[0].set(xlabel="Observed elapsed hours",ylabel="Predicted elapsed hours")
        axs[1].set(xlabel="First held-out jobs (chronological)",ylabel="Elapsed hours")
        axs[2].set(xlabel="Epoch",ylabel="Validation MAE (both targets, h)")
        for ax in axs:ax.legend(fontsize=7)
        fig.suptitle(f"{machine}: synthetic behavior, seed {seed}")
        save(fig,machine+"_reconstruction")
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for machine,g in frame.groupby("machine"):
        axs[0].scatter(g.planned_quantity,g.booked_hours,s=5,alpha=.2,label=machine)
        axs[1].hist(g.truth_queue_wait_hours,bins=30,histtype="step",label=machine)
    axs[0].set(xlabel="Planned quantity (known input)",ylabel="True active processing hours")
    axs[1].set(xlabel="Simulated waiting time (h; not a model input)",ylabel="Jobs")
    for ax in axs:ax.legend()
    save(fig,"simulation_audit")
