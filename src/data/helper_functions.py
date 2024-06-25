import os

from PIL import Image, UnidentifiedImageError

def png_to_jpg(filename):
    """
    Converts a jpg file to a png file located at {filename}
    """
    try:
        im = Image.open(filename)
        rgb_im = im.convert("RGB")
        if filename.endswith("png"):
            jpg_filename = filename.replace(".png", ".jpg")
            rgb_im.save(jpg_filename)
            os.remove(filename)
    except UnidentifiedImageError as uie:
        pass
