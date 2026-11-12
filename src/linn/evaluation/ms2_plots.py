from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(predictions, history, metrics, out):
    out=Path(out); out.mkdir(exist_ok=True)
    colors={"median":"#777777","data_driven":"#0072B2","domain_informed":"#D55E00"}
    def save(fig,name):
        fig.tight_layout()
        for ext in ("png","pdf"): fig.savefig(out/(name+"."+ext),dpi=180)
        plt.close(fig)
    for machine in sorted(predictions.machine.unique()):
        p=predictions[(predictions.machine==machine)&(predictions.split=="test")]
        seed=int(p.seed.min())
        p=p[p.seed==seed]
        fig,axs=plt.subplots(2,3,figsize=(12,7))
        for row,target in enumerate(["elapsed_hours","booked_hours"]):
            g=p[p.target==target]
            for method in ("data_driven","domain_informed"):
                m=g[g.method==method]
                axs[row,0].scatter(m.observed,m.predicted,s=14,alpha=.5,label=method,color=colors[method])
                axs[row,1].hist(m.predicted-m.observed,bins=15,histtype="step",label=method,color=colors[method])
                ordered=np.sort(m.predicted.to_numpy())
                axs[row,2].plot(ordered,np.arange(1,len(ordered)+1)/len(ordered),label=method,color=colors[method])
            observed=np.sort(g[g.method=="data_driven"].observed.to_numpy())
            axs[row,2].plot(observed,np.arange(1,len(observed)+1)/len(observed),label="observed",color="black",linestyle="--")
            limit=max(g.observed.max(),g.predicted.max())
            axs[row,0].plot([0,limit],[0,limit],"k--",lw=1)
            axs[row,0].set(xlabel="Observed hours",ylabel="Predicted hours",title=target)
            axs[row,1].set(xlabel="Prediction minus observed (h)",ylabel="Bookings")
            axs[row,2].set(xlabel="Hours",ylabel="Empirical cumulative fraction")
            for ax in axs[row]: ax.legend(fontsize=7)
        fig.suptitle(f"{machine}: held-out bookings, illustrative seed {seed}")
        save(fig,machine.replace(" ","_")+"_predictions")
        fig,axs=plt.subplots(1,3,figsize=(12,3.7))
        h=history[(history.machine==machine)&(history.seed==seed)]
        for method in ("data_driven","domain_informed"):
            g=h[h.method==method]
            for phase,style in [("train","-"),("validation","--")]:
                axs[0].plot(g.epoch.to_numpy(),g[phase+"_mae_hours"].to_numpy(),style,color=colors[method],label=method+" "+phase)
            axs[1].plot(g.epoch.to_numpy(),g.validation_data_loss.to_numpy(),color=colors[method],label=method)
        g=h[h.method=="domain_informed"]
        for name in ("elapsed_nonnegative","booked_nonnegative","booked_within_elapsed"):
            axs[2].plot(g.epoch.to_numpy(),g["validation_"+name].to_numpy(),label=name)
        axs[0].set(xlabel="Epoch",ylabel="MAE (hours)");axs[1].set(xlabel="Epoch",ylabel="Validation normalized MSE")
        axs[2].set(xlabel="Epoch",ylabel="Unweighted constraint loss")
        for ax in axs: ax.legend(fontsize=7)
        fig.suptitle(machine+": learning curves (first evaluation seed)")
        save(fig,machine.replace(" ","_")+"_learning")
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    m=metrics[(metrics.split=="test")&(metrics.target=="elapsed_hours")]
    machines=sorted(m.machine.unique()); x=np.arange(len(machines))
    for i,method in enumerate(colors):
        g=m[m.method==method].groupby("machine").agg(mae=("MAE","mean"),sd=("MAE","std"),violation=("constraint_violation","mean")).reindex(machines)
        axs[0].bar(x+(i-1)*.25,g.mae,.25,yerr=g.sd.fillna(0),label=method,color=colors[method])
        axs[1].bar(x+(i-1)*.25,100*g.violation,.25,label=method,color=colors[method])
    paired=m[m.method!="median"].pivot(index=["machine","seed"],columns="method",values="MAE")
    improvements=100*(paired.data_driven-paired.domain_informed)/paired.data_driven
    for i,machine in enumerate(machines):
        values=improvements.loc[machine].to_numpy()
        axs[2].scatter(np.full(len(values),i),values,color=colors["domain_informed"])
    axs[0].set(ylabel="Elapsed MAE (h), mean ± seed SD");axs[1].set(ylabel="Any constraint violation (%)")
    axs[2].set(ylabel="Paired MAE improvement (%)");axs[2].axhline(0,color="black",lw=1)
    for ax in axs: ax.set_xticks(x);ax.set_xticklabels(machines)
    axs[0].legend(fontsize=7)
    save(fig,"baseline_comparison")
