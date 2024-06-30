import os

from PIL import Image, UnidentifiedImageError
def png_to_jpg(path):
    """
    Converts a jpg file to a png file located at {filename}
    """
    try:
        im = Image.open(path)
        rgb_im = im.convert("RGB")
        if path.endswith("png"):
            jpg_filename = path.replace(".png", ".jpg")
            rgb_im.save(jpg_filename)
            os.remove(path)
    except UnidentifiedImageError as uie:
        pass


def resize(im: Image, dim: int = 512) -> None:
    """
    - im: image to resize
    - dim: desired image output dimension
    """
    try:
        w, h = im.size
        if w > h:  # case 1, chop off right & left
            margin = (w - h) // 2
            box = (margin, 0, w - margin, h)  # left, top, right, bottom
        else:  # h >= w
            margin = (h - w) // 2
            box = (0, margin, w, h - margin)
        im_ = im.crop(box).resize((dim, dim), Image.Resampling.LANCZOS)

        # create new file path
        im_.save(f"{self.fout}/{filename}")
    except UnidentifiedImageError:
        pass
