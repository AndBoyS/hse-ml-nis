from typing import Union

import numpy as np
from PIL import Image


def feature_extractor_to_numpy(
        feature_extractor,
        x: Union[Image.Image, np.ndarray],
        ) -> np.ndarray:
    """
    Использовать feature_extractor от huggingface и трансформировать аутпут в нампай
    """
    d = feature_extractor(np.array(x))
    x = d['pixel_values'][0]
    x = x.transpose(1, 2, 0)
    return x
