import argparse

from trainer import training


def main():
    parser = argparse.ArgumentParser()
    # project
    parser.add_argument(
        "--project_name", type=str, default="Style-Transfer-for-Anime-Sketches"
    )
    parser.add_argument("--logdir", type=str, default="experiment")
    parser.add_argument("--device", type=str, default="cpu")

    # data
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)

    # model
    parser.add_argument("--dim", type=int, default=16)

    # training
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--beta_1", type=float, default=0.5)
    parser.add_argument("--beta_2", type=float, default=0.99)

    # loss weights
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=100)

    # logger
    parser.add_argument("--upload_artifacts", action="store_true")

    args = parser.parse_args()
    training(args)


if __name__ == "__main__":
    main()
