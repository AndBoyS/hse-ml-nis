from functools import partial

import numpy as np
import gradio as gr

from pkg import models


def detect_glasses(model: str, im: np.ndarray) -> str:
    """
    Detects if a person is wearing glasses in an image.

    Parameters
    ----------
    model : str
        Path to the model to use.
    im : np.ndarray
        The image to detect glasses in.

    Returns
    -------
    float
        The probability of the person wearing glasses.
    """
    m = models.GlassProbaPredictor(model)

    return f'{m.predict_proba(im):.2%}'


def main():
    """Runs the demo."""
    output1 = gr.components.Label(label="Probability of wearing glasses")
    output2 = gr.components.Label(label="Probability of wearing glasses")

    with gr.Blocks(title="Glass Detector") as demo:
        with gr.Tab("Upload"):
            with gr.Row():
                with gr.Column():
                    dropdown = gr.inputs.Dropdown(
                        choices=["microsoft/resnet-50", "microsoft/resnet-101"],
                        label="Model",
                        default="microsoft/resnet-50",
                    )
                    image = gr.components.Image(shape=(224, 224), label="Image", source="upload")
                    gr.Examples(
                        [f"demo_data/{i}.png" for i in range(1, 9)],
                        inputs=image,
                        label="Examples",
                        fn=partial(detect_glasses, model="microsoft/resnet-50"),
                        outputs=output1,
                        examples_per_page=5,
                    )
                with gr.Column():
                    output1.render()
                    button = gr.Button("Detect")
                    button.click(detect_glasses, [dropdown, image], outputs=output1)

        with gr.Tab("Take picture"):
            with gr.Row():
                with gr.Column():
                    dropdown = gr.inputs.Dropdown(
                        choices=["microsoft/resnet-50", "microsoft/resnet-101"],
                        label="Model",
                        default="microsoft/resnet-50",
                    )
                    webcam = gr.components.Image(shape=(224, 224), label="Image", source="webcam")
                with gr.Column():
                    output2.render()
                    button = gr.Button("Detect")
                    button.click(detect_glasses, [dropdown, webcam], outputs=output2)

    demo.launch()


if __name__ == "__main__":
    main()
