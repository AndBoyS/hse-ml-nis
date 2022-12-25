import warnings
from typing import List

from PIL import Image
import torch
from torch import nn
import torchvision.transforms as T
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import pytorch_lightning as pl

from .utils import feature_extractor_to_numpy


class GlassProbaPredictorTrained(pl.LightningModule):
    def __init__(
            self,
            hf_model_name: str,
            learning_rate: float = 1e-2,
            warmup_steps: int = 2,
            milestones: List[int] = [1000, 2000, 3000],
            gamma: float = 0.1,
            train_only_last_layer: bool = False,
    ):
        """
        hf_model_name: str
            Название предобученной модели на huggingface (например 'microsoft/resnet-50')
        milestones: List[int]
            Список эпох для learning rate decay
        gamma: float
            Множитель в learning rate decay
        train_only_last_layer: bool
            Заморозить ли все веса кроме последнего слоя
        """
        super().__init__()
        self.model_name = hf_model_name
        self._init_model(hf_model_name, train_only_last_layer)
        self.loss = nn.BCELoss()
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.milestones = milestones
        self.save_hyperparameters()

    def _init_model(self, model_name, train_only_last_layer):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.body = AutoModelForImageClassification.from_pretrained(model_name)

        # Замораживаем все кроме выходного слоя
        if train_only_last_layer:
            for param in self.parameters():
                param.requires_grad = False

        # Сетап для предобученных моделей microsoft/resnet-xx
        in_features = self.body.classifier[-1].in_features

        self.body.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1),
        )

    def predict_proba(self, im):
        with torch.no_grad():
            return self.forward(im).sigmoid().item()

    def forward(self, x):
        if isinstance(x, Image.Image):
            x = feature_extractor_to_numpy(self.feature_extractor, x)
            x = T.ToTensor()(x)[None, ...]
        elif not isinstance(x, torch.Tensor):
            raise ValueError(f'Input should be either PIL.Image or torch.Tensor from dataset dataloader, got {type(x)}')
            
        x = self.body(x).logits 
        return x

    def compute_loss_on_batch(self, batch):
        im = batch['image']
        label = batch['label']

        pred = self.forward(im).sigmoid()
        pred = pred.squeeze()
        label = label.float()
        loss = self.loss(pred, label)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss_on_batch(batch)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss_on_batch(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss_on_batch(batch)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=1e-3,
        )
        return optimizer
    
    # learning rate warm-up + learning rate decay
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        current_lr = self.get_warmup_lr(
            self.trainer.global_step,
            self.hparams.learning_rate,
        )

        current_lr = self.get_decay_lr(
            self.trainer.global_step,
            current_lr,
        )
        
        self.set_optimizer_lr(optimizer, current_lr)
        self.log("lr", current_lr)
        
    def get_warmup_lr(self, cur_step, base_lr):
        # Linear warmup 
        current_lr = base_lr
        if cur_step < self.warmup_steps:
            lr_scale = min(1.0, float(cur_step + 1) / self.warmup_steps)
            current_lr = lr_scale * current_lr
            
        return current_lr

    def get_decay_lr(self, cur_step, cur_lr):
        new_lr = cur_lr
        
        factor = self._get_decay_lr_factor(cur_step)
        new_lr = cur_lr * self.gamma ** factor
        return new_lr
    
    def _get_decay_lr_factor(self, cur_step):
        # В какой степени gamma умножается на lr
        
        factor = 0
        for milestone in self.milestones:
            if milestone < cur_step:
                factor += 1
            else: 
                break
            
        return factor
    
    @staticmethod
    def set_optimizer_lr(optimizer, lr):
        for pg in optimizer.param_groups:
            pg["lr"] = lr


class GlassProbaPredictor(nn.Module):
    """
    (DEPRECATED)
    Загружает предобученную на ImageNet-1k (или на другом датасете, содержащем очки) модель с huggingface
    Предсказывает вероятность нахождения на фото очков
    """

    glasses_class: int
    feature_extractor: nn.Module
    model: nn.Module

    def __init__(
            self,
            hf_model_name: str,
            glasses_class_name: str = 'sunglasses, dark glasses, shades',
    ):
        """
        hf_model_name: str
            Название предобученной модели на huggingface (например 'microsoft/resnet-50')
        glasses_class_name: str
            Название класса очков в датасете, на котором обучалась модель (по умолчанию используется название из ImageNet-1k)
        """

        super().__init__()

        warnings.warn('This module is deprecated')

        self.model_name = hf_model_name
        self._init_model(hf_model_name)
        self.glasses_class = self.model.config.label2id[glasses_class_name]

    def _init_model(self, model_name):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)

    def forward(self, x):
        inputs = self.feature_extractor(x, return_tensors="pt")
        pred = self.model(**inputs)
        return pred

    def predict_proba(self, x):
        """
        Предсказать вероятность нахождения на фото очков
        """
        with torch.no_grad():
            pred = self(x).logits.softmax(1)

        glass_prob = pred[0, self.glasses_class].item()
        return glass_prob
    