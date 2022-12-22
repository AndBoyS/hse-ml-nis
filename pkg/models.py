import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

from pkg.utils import prune_linear_layer


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
            output_layer_pruned: bool = True,
    ):
        """
        hf_model_name: str
            Название предобученной модели на huggingface (например 'microsoft/resnet-50')
        glasses_class_name: str
            Название класса очков в датасете, на котором обучалась модель (по умолчанию используется название из ImageNet-1k)
        output_layer_pruned: bool
            Оставить только один выходной нейрон для ускорения инференса
        """

        super().__init__()

        self.model_name = hf_model_name
        self.output_layer_pruned = output_layer_pruned
        self._init_model(hf_model_name)
        self.glasses_class = self.model.config.label2id[glasses_class_name]
        self._prune_output_layer()

    def _init_model(self, model_name):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)

    def _prune_output_layer(self):
        if not self.output_layer_pruned:
            return

        prune_linear_layer(self.model, neurons_to_keep=(self.glasses_class,))
        self.glasses_class = 0

    def forward(self, x):
        inputs = self.feature_extractor(x, return_tensors="pt")
        pred = self.model(**inputs)
        return pred

    def predict_proba(self, x):
        """
        Предсказать вероятность нахождения на фото очков
        """
        with torch.no_grad():
            pred = self(x).logits.sigmoid()

        glass_prob = pred[0, self.glasses_class].item()
        return glass_prob
