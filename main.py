from pathlib import Path

import numpy as np
from PIL import Image
import click

from pkg import models


default_model_name = 'microsoft/resnet-50'


@click.command()
@click.argument("folder_path")
@click.option("--model_name", default=default_model_name, required=False,
              help=f'Huggingface model name, default={default_model_name}')
def main(folder_path: str, model_name: str):
    """
    folder_path: Path to the folder with images to be inferenced on (images can be of types jpg, jpeg, png)
    """

    folder_path = Path(folder_path)

    model = models.GlassProbaPredictor(model_name)

    click.echo(f'Glass probabilities for images:')

    for fp in folder_path.glob('*'):
        if fp.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        im = Image.open(fp)
        prob = model.predict_proba(np.array(im))
        click.echo(f'{fp}: {prob:.2f}')


if __name__ == '__main__':
    main()
