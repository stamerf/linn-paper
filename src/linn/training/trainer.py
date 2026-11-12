"""Generic trainer. It imports neither a workstation class nor Company 2 columns."""
import random
import numpy as np
import torch
from .losses import masked_mse


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


class Trainer:
    def __init__(self, model, constraints, context, learning_rate=1e-3, constraint_weight=0.1, weight_decay=0.):
        if constraint_weight < 0:
            raise ValueError("Constraint weight must be nonnegative")
        self.model, self.constraints, self.context = model, constraints, context
        self.weight = constraint_weight
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def prediction(self, features):
        raw = self.model(features)
        return raw * torch.as_tensor(self.context["target_scale"], dtype=raw.dtype, device=raw.device)

    def step(self, batch, train=True):
        self.model.train(train)
        with torch.set_grad_enabled(train):
            prediction = self.prediction(batch["features"])
            data = masked_mse(prediction, batch["targets"], batch["target_mask"], self.context["target_scale"])
            terms = self.constraints.compute(prediction, batch.get("constraint_inputs", {}), self.context)
            for name, term in terms.items():
                if term.ndim != 0 or not torch.isfinite(term):
                    raise ValueError(f"Invalid constraint loss: {name}")
            constraint = sum(terms.values(), prediction.sum()*0)
            total = data + self.weight*constraint
            if not torch.isfinite(total):
                raise ValueError("Nonfinite total loss")
            if train:
                self.optimizer.zero_grad()
                total.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                    raise ValueError("Nonfinite gradient")
                self.optimizer.step()
                if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                    raise ValueError("Nonfinite model parameter")
        return {"total_loss": float(total.detach()), "data_loss": float(data.detach()),
                "constraint_loss": float(constraint.detach()),
                **{name: float(value.detach()) for name, value in terms.items()}}

    def epoch(self, loader, train=True):
        """Aggregate data loss by observed entries and constraints by sample count."""
        samples, observed, data_sum, constraint_sums = 0, 0, 0., {}
        for batch in loader:
            log = self.step(batch, train=train)
            n, n_observed = len(batch["features"]), int(batch["target_mask"].sum())
            samples += n
            observed += n_observed
            data_sum += log["data_loss"]*n_observed
            for name, value in log.items():
                if name not in ("total_loss", "data_loss"):
                    constraint_sums[name] = constraint_sums.get(name, 0.) + value*n
        if samples == 0:
            raise ValueError("Empty loader")
        result = {name: value/samples for name, value in constraint_sums.items()}
        result["data_loss"] = data_sum/observed if observed else 0.
        result["total_loss"] = result["data_loss"] + self.weight*result["constraint_loss"]
        return result

    def fit(self, train_loader, validation_loader, epochs):
        """Generic epoch loop. Callers own split construction and model selection."""
        if epochs < 1:
            raise ValueError("epochs must be positive")
        return [{"epoch": epoch+1, "train": self.epoch(train_loader),
                 "validation": self.epoch(validation_loader, train=False)} for epoch in range(epochs)]
