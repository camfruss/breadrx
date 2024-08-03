from src.ml.architecture import ConvolutionalNN, ImageDataset

from argparse import ArgumentParser
import os
from PIL import Image
from time import time
import torch
from torch import nn

def main():
    parser = ArgumentParser(
        prog="breadrx",
        description="To determining whether bread is under, over, or well-proofed"
    )
    parser.add_argument("-f", "--file", required=True, type=str, help="JPG crumb image")
    # TODO: add more options
    # parser.add_argument("-llm", "--large-language-model", action="store_true", help="")

    args = parser.parse_args()

    im_path = args.file
    print(im_path)
    if os.path.exists(im_path):
        try:
            image = Image.open(im_path)
        except Exception:
            raise Exception("Error loading image file")
    else:
        raise FileNotFoundError

    model = ConvolutionalNN()
    path = os.path.join(os.path.dirname(__file__), "breadrx-cnn.pth")

    ts_load = time()
    model.load_state_dict(torch.load(path))
    tf_load = time()

    model.eval()

    dim = 64  # TODO: parameterize
    transform = ImageDataset.s_get_transforms(dim)
    image = transform(image)

    ts_inf = time()
    pred = model(image.unsqueeze(0))
    tf_inf = time()

    probs = nn.Softmax(dim=1)(pred)
    pred = pred.argmax(1).item()

    mapping = {
        0: "under-proofed",
        1: "over-proofed",
        2: "well-proofed"
    }

    print(f"{mapping.get(pred)} with probability of {probs[0, pred]:.3f}")
    print(f"> model load time: {tf_load - ts_load:.3f} seconds")
    print(f"> inference time: {tf_inf - ts_inf:.3f} seconds")


if __name__ == "__main__":
    main()
