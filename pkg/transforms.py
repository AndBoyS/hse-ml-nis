import numpy as np
import torch


class CutOut:
    def __init__(self,
                 probability=0.5,
                 num_rand=(1, 3),
                 box_width_rate=(0.1, 0.3),
                 box_height_rate=(0.1, 0.3),
                 gray_scale=(0, 255)):
        """

        :param probability:
            Вероятность применения аугментации
        :param num_rand:
            Разброс количества возможных катаутов
        :param box_width_rate:
            Разброс в доле от ширины фото
        :param box_height_rate:
            Разброс в доле от высоты фото
        :param gray_scale:
            Разброс значений цвета катаута
        """

        self.probability = probability
        self.num_rand = num_rand
        self.box_width_rate = np.array(box_width_rate)
        self.box_height_rate = np.array(box_height_rate)
        self.gray_scale = gray_scale

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if np.random.random() < self.probability:
            h, w = img.shape[-2:]
            box_w_range = (w * self.box_width_rate).round()
            box_h_range = (h * self.box_height_rate).round()

            num_rand = np.random.randint(*self.num_rand)
            for _ in range(num_rand):

                cut_w_len = np.random.randint(*box_w_range)
                cut_h_len = np.random.randint(*box_h_range)
                cut_h_start, cut_w_start = np.random.randint(0, h - cut_h_len), np.random.randint(0, w - cut_w_len)
                cut_h_slice = slice(cut_h_start, cut_h_start + cut_h_len)
                cut_w_slice = slice(cut_w_start, cut_w_start + cut_w_len)

                img[..., cut_h_slice, cut_w_slice] = np.random.randint(*self.gray_scale)

        return img
