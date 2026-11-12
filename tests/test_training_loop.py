import pytest
import torch
from torch.utils.data import DataLoader
from linn.models.neural_model import NeuralRegressor
from linn.constraints.base_constraint import NoConstraints
from linn.training.trainer import Trainer, seed_everything


def test_epoch_validation_does_not_update_and_mask_weighting():
    seed_everything(1)
    model = NeuralRegressor(1,2,())
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    rows = [{"features":torch.zeros(1), "targets":torch.tensor(y), "target_mask":torch.tensor(m)}
            for y,m in [([1.,9.],[True,False]),([2.,3.],[True,True])]]
    loader = DataLoader(rows,batch_size=1)
    trainer = Trainer(model,NoConstraints(),{"target_scale":[1,1]})
    before = [p.clone() for p in model.parameters()]
    result = trainer.epoch(loader,train=False)
    assert result["data_loss"] == pytest.approx(14/3)
    assert all(torch.equal(p,q) for p,q in zip(before,model.parameters()))
    history = trainer.fit(loader,loader,epochs=2)
    assert len(history) == 2 and "validation" in history[0]
    with pytest.raises(ValueError,match="Empty"):
        trainer.epoch([])
