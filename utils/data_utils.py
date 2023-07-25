"""
Code adopted from pix2pixHD (https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py)
"""
from pathlib import Path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename: Path):
    return any(str(filename).endswith(extension) for extension in IMG_EXTENSIONS)


from pathlib import Path

def make_dataset(dir: str):
    dir_path = Path(dir)
    images = []
    assert dir_path.is_dir(), '%s is not a valid directory' % dir_path
    for fname in dir_path.glob("*"):
        if is_image_file(fname):
            path = dir_path / fname
            images.append(path)
    return images

