import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


class GlassProbaPredictor(nn.Module):
    """
    Загружает предобученную на ImageNet-1k (или на другом датасете, содержащем очки) модель с huggingface
    Предсказывает вероятность нахождения на фото очков
    """

    glasses_class = None  # Класс очков

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

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(hf_model_name)
        self.model = AutoModelForImageClassification.from_pretrained(hf_model_name)
        self.glasses_class = self.model.config.label2id[glasses_class_name]

    def forward(self, x):
        inputs = self.feature_extractor(x, return_tensors="pt")
        logits = self.model(**inputs).logits
        return logits

    def predict_proba(self, x):
        """
        Предсказать вероятность нахождения на фото очков
        """
        with torch.no_grad():
            pred = self(x).softmax(1)

        glass_prob = pred[0, self.glasses_class].item()
        return glass_prob