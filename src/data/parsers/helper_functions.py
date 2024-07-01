import os
from PIL import Image, UnidentifiedImageError


def filter_links(df, patterns):
    query_pattern = "|".join(patterns)
    df = df[df["image_link"].str.contains(query_pattern)]
    return df


def png_to_jpg(path):
    """ Converts a jpg file to a png file located at {filename} """
    try:
        im = Image.open(path)
        rgb_im = im.convert("RGB")
        if path.endswith("png"):
            jpg_filename = path.replace(".png", ".jpg")
            rgb_im.save(jpg_filename)
            os.remove(path)
    except UnidentifiedImageError:
        pass


def resize(path, new_path: str = None, dim: int = 512) -> None:
    """
    - path: directory of images to resize
    - new_path: new location of resized images.
        - warning: if not provided, all images in {path} will be overwritten
    - dim: desired image output dimension
        - all images are cropped to a square
    """
    if new_path:  # create new path
        os.makedirs(new_path, exist_ok=True)

    for fp in os.listdir(path):
        filepath = os.path.join(path, fp)
        try:
            im = Image.open(filepath)
            w, h = im.size
            if w > h:  # case 1, chop off right & left
                margin = (w - h) // 2
                box = (margin, 0, w - margin, h)  # left, top, right, bottom
            else:  # h >= w
                margin = (h - w) // 2
                box = (0, margin, w, h - margin)
            im_ = im.crop(box).resize((dim, dim), Image.Resampling.LANCZOS)

            # create new file path
            if new_path:
                filepath = os.path.join(new_path, fp)
            im_.save(filepath)
        except UnidentifiedImageError:
            pass
