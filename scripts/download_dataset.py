import subprocess
from pathlib import Path
from typing import Union, Tuple
import urllib.request

import gdown


def download_meglass(data_dir: Union[Path, str]) -> Tuple[Path, Path]:
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    dataset_archive_fp = data_dir / "data.zip"

    dataset_link = 'https://drive.google.com/u/0/uc?id=1V0c8p6MOlSFY5R-Hu9LxYZYLXd8B8j9q&export=download'
    dataset_dir = data_dir / 'MeGlass_120x120'  # Как называется папка, которая будет распакована из скачанного архива

    if not dataset_archive_fp.exists():
        gdown.download(dataset_link, str(dataset_archive_fp), quiet=False)

    if not dataset_dir.exists():
        subprocess.run(['unzip', '-d', str(data_dir), str(dataset_archive_fp)])

    labels_fp = dataset_dir / 'meta.txt'
    labels_link = 'https://github.com/cleardusk/MeGlass/raw/master/meta.txt'
    urllib.request.urlretrieve(labels_link, labels_fp)

    return dataset_dir, labels_fp


