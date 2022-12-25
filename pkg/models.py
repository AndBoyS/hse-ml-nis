import warnings
from typing import List
from collections import Counter

import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import pytorch_lightning as pl


class GlassProbaPredictor(nn.Module):
    """
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


class GlassProbaPredictorTrained(pl.LightningModule):
    def __init__(
            self,
            hf_model_name: str,
            learning_rate: float = 1e-3,
            #warmup_steps: int = 10000,
            warmup_steps: int = 10,
            #milestones: List[int] = [20000, 25000, 30000],
            milestones: List[int] = [20, 30, 40],
            gamma: float = 0.1,
    ):
        """
        hf_model_name: str
            Название предобученной модели на huggingface (например 'microsoft/resnet-50')
        milestones: List[int]
            Список эпох для learning rate decay
        gamma: float
            Множитель в learning rate decay
        """
        super().__init__()
        self.model_name = hf_model_name
        self._init_model(hf_model_name)
        self.loss = nn.BCELoss()
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.milestones = milestones
        # self.milestones_counter = Counter(milestones)
        self.save_hyperparameters()

    def _init_model(self, model_name):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.body = AutoModelForImageClassification.from_pretrained(model_name)

        # Замораживаем все кроме выходного слоя
        for param in self.parameters():
            param.requires_grad = False

        # Сетап для предобученных моделей microsoft/resnet-xx
        in_features = self.body.classifier[-1].in_features

        self.body.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.feature_extractor(x)
        return self.body(x).logits

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=self.milestones,
            gamma=self.gamma,
        )
        
        scheduler = WarmUpScheduler(
            optimizer, 
            scheduler,
            len_loader=1,
            warmup_steps=self.warmup_steps,
            warmup_start_lr=1e-7,
            warmup_mode='linear',
        )
        
        #scheduler = create_lr_scheduler_with_warmup(
        #    scheduler,
        #    warmup_start_value=1e-7,
        #    warmup_duration=self.warmup_steps,
        #)
        
        scheduler_config = {
            'scheduler': scheduler,
            'name': 'multisteplr',
        }
        return [optimizer], [scheduler_config]
    
    # learning rate warm-up + learning rate decay
    #def optimizer_step(
    #    self,
    #    epoch,
    #    batch_idx,
    #    optimizer,
    #    optimizer_idx,
    #    optimizer_closure,
    #    on_tpu=False,
    #    using_native_amp=False,
    #    using_lbfgs=False,
    #):
    #    # update params
    #    optimizer.step(closure=optimizer_closure)
#
    #    current_lr = self.get_warmup_lr(
    #        self.trainer.global_step,
    #        self.hparams.learning_rate,
    #        self.warmup_steps,
    #    )
    #    
    #    #current_lr = self.get_decay_lr(
    #    #    self.trainer.global_step,
    #    #    current_lr,
    #    #    self.milestones_counter,
    #    #    self.gamma,
    #    #)
    #    
    #    self.set_optimizer_lr(optimizer, current_lr)
    #    self.log("lr", current_lr)
        
    @staticmethod
    def get_warmup_lr(cur_step, base_lr, warmup_steps):
        # Linear warmup 
        current_lr = base_lr
        if cur_step < warmup_steps:
            lr_scale = min(1.0, float(cur_step + 1) / warmup_steps)
            current_lr = lr_scale * current_lr
            
        return current_lr

    @staticmethod
    def get_decay_lr(cur_step, cur_lr, milestones, gamma):
        if cur_step not in milestones:
            return cur_lr
        return cur_lr * gamma ** milestones[cur_step]
    
    @staticmethod
    def set_optimizer_lr(optimizer, lr):
        for pg in optimizer.param_groups:
            pg["lr"] = lr
    