from parsers.helper_functions import resize
from parsers.imgur_parser import ImgurParser
from parsers.reddit_parser import RedditParser
from parsers.torrent_parser import TorrentParser


def main():
    tp = TorrentParser("./raw/uncompressed")
    df = tp.all()

    rp = RedditParser(df, path="./tmp/reddit")
    # rp = RedditParser(filepath="./raw/uncompressed/torrent_out.csv", path="./tmp/reddit")
    rp.parse(limit=5)
    rp.download(politeness=3)

    # ip = ImgurParser(df, path="./tmp/imgur")
    ip = ImgurParser(filepath="./raw/uncompressed/torrent_out.csv", path="./tmp/imgur")
    ip.parse(limit=5)
    ip.download()

    resize("./tmp/imgur")


if __name__ == "__main__":
    main()
