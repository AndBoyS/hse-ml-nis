from pathlib import Path
from typing import Union

import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from pkg.models import GlassProbaPredictorTrained
from pkg.dataset import MeGlassDataset


def train(
        dataset_fp: Union[Path, str],
        label_fp: Union[Path, str],
        model_name: str = 'microsoft/resnet-18',
        ):

    dataset = MeGlassDataset(dataset_fp, label_fp)
    dataset_len = len(dataset)

    idx = range(dataset_len)
    train_idx, test_idx = train_test_split(idx, test_size=0.01, random_state=42)
    val_idx, test_idx = train_test_split(idx, test_size=0.5, random_state=42)

    train_dataset = MeGlassDataset(dataset_fp, label_fp, idx=train_idx)
    val_dataset = MeGlassDataset(dataset_fp, label_fp, idx=val_idx)
    test_dataset = MeGlassDataset(dataset_fp, label_fp, idx=test_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset)

    model = GlassProbaPredictorTrained(model_name)
    trainer = pl.Trainer(
        limit_train_batches=100,
        max_epochs=1,
        auto_lr_find=True,
    )
    trainer.fit(
        model,
        train_loader,
        val_loader,
    )

    trainer.test(model, dataloaders=test_loader)
