from pathlib import Path

import numpy as np
from PIL import Image
import click

from pkg import models


default_model_name = 'microsoft/resnet-50'


@click.command()
@click.argument("folder_path")
def main(folder_path: str):
    """
    folder_path: Path to the folder with images to be inferenced on (images can be of types jpg, jpeg, png)
    """

    folder_path = Path(folder_path)
    weights_dir = Path('weights')

    model = models.GlassProbaPredictorTrained.load_from_checkpoint(
        weights_dir / "resnet18_finetuned.ckpt",
        hf_model_name='microsoft/resnet-18',
    )
    model.eval()

    click.echo(f'Glass probabilities for images:')

    for fp in folder_path.glob('*'):
        if fp.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        im = Image.open(fp).convert('RGB')
        prob = model.predict_proba(im)
        click.echo(f'{fp}: {prob:.2f}')


if __name__ == '__main__':
    main()
