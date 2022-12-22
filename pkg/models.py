import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoModelForObjectDetection


class GlassProbaPredictor(nn.Module):
    """
    Загружает предобученную на ImageNet-1k (или на другом датасете, содержащем очки) модель с huggingface
    Предсказывает вероятность нахождения на фото очков
    """

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
        glasses_class_name
            Название класса очков в датасете, на котором обучалась модель (по умолчанию используется название из ImageNet-1k)
        """

        super().__init__()

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
            pred = self(x).logits.sigmoid()

        glass_prob = pred[0, self.glasses_class].item()
        return glass_prob


class GlassBboxPredictor(GlassProbaPredictor):
    """
    Загружает предобученную на COCO 2017 (или на другом датасете, содержащем очки) модель с huggingface
    Предсказывает вероятность нахождения на фото очков/баундинг бокс очков, если модель их выдает
    """

    def __init__(
            self,
            hf_model_name: str,
            glasses_class_name: str = 'sunglasses, dark glasses, shades',
    ):
        super().__init__(hf_model_name, glasses_class_name)

    def _init_model(self, model_name):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)

    def predict_bbox(self, x):
        """
        Предсказать баундинг бокс очков
        """
        with torch.no_grad():
            pred = self(x)

        bbox = getattr(pred, 'pred_boxes', None)
        assert bbox is not None, f'{self.model_name} не предсказывает баундинг боксы'
        return bbox


