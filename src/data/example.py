from parse import DataIngestor


def main():
    path = "./subreddits23/uncompressed"
    subreddits = ["Sourdough", "Breadit"]
    di = DataIngestor(path, subreddits)
    di.save("./subreddits", "from_scratch.csv")


if __name__ == "__main__":
    main()
