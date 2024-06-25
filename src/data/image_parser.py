from image_sources import RedditImages, ImgurImages

import os
from PIL import Image, UnidentifiedImageError

class ImageParser:

    def __init__(self, fin: str, fout: str, df):
        """

        Parameters
        ----------
        - fin: folder containing the images to parse
            e.g.: "./filtered/crumb"
        - fout: desired saved images folder. creates folder if does not exist
        - df
            - must contain a column "image_link"
        """
        self.fin = fin.removesuffix("/")
        self.fout = fout.removesuffix("/")
        self.df = df

        if not os.path.exists(fout):
            os.mkdir(fout)

    """
    Downloads images from the link provided in {image_link} using the parser specified in {source_map}
    """
    def download_images(self):
        source_map = {
            "reddit": (["i.redd.it", "reddit.com"], RedditImages),
            "imgur": (["imgur"], ImgurImages)
        }

        for k, (v, cls) in source_map.items():
            df = self.df[self.df["image_link"].str.contians("|".join(v))]
            cls(df, self.fin, self.fout).parse()

    def resize_dir(self, dim: int = 512):
        """
        Resizes all images in {dir_} to square images of equal height and width of {dim}

        Parameters
        ----------
        - dim: desired image output dimension
            TODO: is a hyperparameter
        """

        for filename in os.listdir(self.fin):
            try:
                im = Image.open(f"{self.fin}/{filename}")

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
