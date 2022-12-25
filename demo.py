from functools import partial

import numpy as np
import gradio as gr
from PIL import Image

from pkg import models

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def detect_glasses(im: np.ndarray) -> str:
    """
    Detects if a person is wearing glasses in an image.

    Parameters
    ----------
    im : np.ndarray
        The image to detect glasses in.

    Returns
    -------
    str
        The probability of the person wearing glasses.
    """
    im = Image.fromarray(im)

    m = models.GlassProbaPredictorTrained.load_from_checkpoint(
        "weights/resnet18_finetuned.ckpt",
        hf_model_name='microsoft/resnet-18',
    )
    m.eval()

    return f'{m.predict_proba(im):.2%}'


def main():
    """Runs the demo."""
    output1 = gr.components.Label(label="Probability of wearing glasses")
    output2 = gr.components.Label(label="Probability of wearing glasses")

    with gr.Blocks(title="Glass Detector") as demo:
        with gr.Tab("Upload"):
            with gr.Row():
                with gr.Column():
                    image = gr.components.Image(shape=(224, 224), label="Image", source="upload")
                    gr.Examples(
                        [f"demo_data/{i}.png" for i in range(1, 9)],
                        inputs=image,
                        label="Examples",
                        fn=partial(detect_glasses),
                        outputs=output1,
                        examples_per_page=5,
                    )
                with gr.Column():
                    output1.render()
                    button = gr.Button("Detect")
                    button.click(detect_glasses, [image], outputs=output1)

        with gr.Tab("Take picture"):
            with gr.Row():
                with gr.Column():
                    webcam = gr.components.Image(shape=(224, 224), label="Image", source="webcam")
                with gr.Column():
                    output2.render()
                    button = gr.Button("Detect")
                    button.click(detect_glasses, [webcam], outputs=output2)

    demo.launch()


if __name__ == "__main__":
    main()
