"""Bounded validation-only tuning and early stopping. No test argument exists here."""
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from linn.models.neural_model import NeuralRegressor
from linn.training.trainer import Trainer, seed_everything


def predict(trainer, dataset):
    trainer.model.eval()
    with torch.no_grad():
        p = trainer.prediction(dataset.x).cpu().numpy()
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite prediction")
    return p


def train_candidate(train, validation, spec, candidate, weight, seed, settings):
    seed_everything(seed)
    model = NeuralRegressor(train.x.shape[1], train.y.shape[1], candidate["hidden_dims"])
    trainer = Trainer(model, spec.constraints, spec.context, candidate["learning_rate"], weight,
                      candidate["weight_decay"])
    loader = DataLoader(train, batch_size=candidate["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    train_eval = DataLoader(train, batch_size=len(train))
    val_eval = DataLoader(validation, batch_size=len(validation))
    best, best_epoch, state, stale, history = float("inf"), 0, None, 0, []
    scale = np.asarray(spec.context["target_scale"])
    for epoch in range(1, settings["max_epochs"]+1):
        trainer.epoch(loader)
        logs = {"epoch": epoch}
        for name, dataset, batches in [("train",train,train_eval),("validation",validation,val_eval)]:
            logs.update({name+"_"+k:v for k,v in trainer.epoch(batches,train=False).items()})
            p = predict(trainer,dataset)
            errors = p-dataset.y.numpy()
            observed = dataset.mask.numpy()
            if not observed.any():
                raise ValueError("Training/validation diagnostics require observed labels")
            logs[name+"_mae_hours"] = float(np.abs(errors[observed]).mean())
            logs[name+"_rmse_hours"] = float(np.sqrt(np.square(errors[observed]).mean()))
            logs[name+"_normalized_mae"] = float(np.abs((errors/scale)[observed]).mean())
        history.append(logs)
        score = logs["validation_normalized_mae"]
        if score < best-settings["min_delta"]:
            best, best_epoch, state, stale = score, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    model.load_state_dict(state)
    return trainer, history, {"validation_score": best, "best_epoch": best_epoch, "epochs_run": len(history)}


def tune(train, validation, spec, settings):
    trials = []
    for i,candidate in enumerate(settings["candidates"]):
        for seed in settings["tuning_seeds"]:
            _,_,info = train_candidate(train,validation,spec,candidate,0.,seed,settings)
            trials.append({"stage":"architecture", "candidate":i, "weight":0., "seed":seed, **info})
    scores = {i:np.mean([r["validation_score"] for r in trials if r["candidate"]==i])
              for i in range(len(settings["candidates"]))}
    best_i = min(scores,key=lambda i:(scores[i],i))
    candidate = settings["candidates"][best_i]
    for weight in settings["constraint_weights"]:
        for seed in settings["tuning_seeds"]:
            _,_,info = train_candidate(train,validation,spec,candidate,weight,seed,settings)
            trials.append({"stage":"constraint_weight", "candidate":best_i,"weight":weight,"seed":seed,**info})
    weight_scores = {w:np.mean([r["validation_score"] for r in trials if r["stage"]=="constraint_weight" and r["weight"]==w])
                     for w in settings["constraint_weights"]}
    best_weight = min(weight_scores,key=lambda w:(weight_scores[w],w))
    return {"candidate":candidate,"candidate_index":best_i,"constraint_weight":best_weight},trials
