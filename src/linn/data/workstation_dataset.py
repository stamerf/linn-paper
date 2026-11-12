"""Train-only feature fitting; target masks are explicit and never passed to constraints."""
import numpy as np
import inspect
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset

TARGETS = ("elapsed_hours", "booked_hours")


def features(frame, numeric_columns=()):
    x = frame[["product", "operation"]].fillna("__missing__").copy()
    hour = frame.start.dt.hour + frame.start.dt.minute / 60
    x["hour_sin"] = np.sin(2*np.pi*hour/24)
    x["hour_cos"] = np.cos(2*np.pi*hour/24)
    x["weekday_sin"] = np.sin(2*np.pi*frame.start.dt.dayofweek/7)
    x["weekday_cos"] = np.cos(2*np.pi*frame.start.dt.dayofweek/7)
    for column in numeric_columns:
        if column in x or column in TARGETS or column in ("end", "record_id", "order_id", "split"):
            raise ValueError(f"Forbidden or duplicate additional feature: {column}")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite additional feature: {column}")
        x[column] = values
    return x


class FeatureEncoder:
    def __init__(self, numeric_columns=()):
        self.numeric_columns = tuple(numeric_columns)
        if len(set(self.numeric_columns)) != len(self.numeric_columns):
            raise ValueError("Duplicate numeric feature names")

    def fit(self, train):
        if train.empty or not train["split"].eq("train").all():
            raise ValueError("Fit requires a nonempty training split only")
        dense_option = "sparse_output" if "sparse_output" in inspect.signature(OneHotEncoder).parameters else "sparse"
        transforms = [
            ("categories", OneHotEncoder(handle_unknown="ignore", **{dense_option: False}), ["product", "operation"]),
            ("calendar", StandardScaler(), ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"])]
        if self.numeric_columns:
            transforms.append(("numeric", StandardScaler(), list(self.numeric_columns)))
        self.transformer = ColumnTransformer(transforms)
        self.transformer.fit(features(train, self.numeric_columns))
        self.fitted_ids = train.record_id.tolist()
        return self

    def transform(self, frame):
        x = self.transformer.transform(features(frame, self.numeric_columns)).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite features")
        return x

    def metadata(self):
        metadata = {"fitted_record_ids": self.fitted_ids,
                "feature_names": self.transformer.get_feature_names_out().tolist(),
                "calendar_mean": self.transformer.named_transformers_["calendar"].mean_.tolist(),
                "calendar_scale": self.transformer.named_transformers_["calendar"].scale_.tolist()}
        if self.numeric_columns:
            metadata.update({"numeric_columns": list(self.numeric_columns),
                             "numeric_mean": self.transformer.named_transformers_["numeric"].mean_.tolist(),
                             "numeric_scale": self.transformer.named_transformers_["numeric"].scale_.tolist()})
        return metadata


class WorkstationDataset(Dataset):
    def __init__(self, frame, encoder, target_mask=None):
        self.x = torch.from_numpy(encoder.transform(frame))
        y = frame[list(TARGETS)].to_numpy(dtype=np.float32)
        mask = np.isfinite(y) if target_mask is None else np.asarray(target_mask, dtype=bool)
        if mask.shape != y.shape or not np.isfinite(y[mask]).all():
            raise ValueError("Invalid target mask or observed target")
        self.y = torch.from_numpy(np.where(mask, y, 0).astype(np.float32))
        self.mask = torch.from_numpy(mask)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"features": self.x[index], "targets": self.y[index],
                "target_mask": self.mask[index], "constraint_inputs": {}}
