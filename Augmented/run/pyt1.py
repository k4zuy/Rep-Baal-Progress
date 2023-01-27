import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="notag", type=str)
    return parser.parse_args()

def main():
    
    args = parse_args()
    hyperparams = vars(args)
    tag = hyperparams["tag"]

    print(f"tag: {tag}")


if __name__ == "__main__":
    main()