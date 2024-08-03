from .architecture import ConvolutionalNN

from argparse import ArgumentParser
import os
import torch


def demo():
    print(1)


def main():
    parser = ArgumentParser(
        prog="breadrx",
        description="desc"
    )
    parser.add_argument("-f", "--file", required=True, type=str, help="image")
    parser.add_argument("-llm", "--large-language-model", action="store_true", help="")

    model = ConvolutionalNN()
    path = os.path.join(os.path.dirname(__file__), "breadrx-cnn.pth")
    model.load_state_dict(torch.load(path))
    model.eval()


    # transforms
    # run through model
    # print result



if __name__ == "__main__":
    demo()
    # main()
