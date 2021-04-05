import argparse
import os
import pathlib
import re

from hanziconv import HanziConv
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lyrics dataset preparation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data",
        type=str,
        help="Path to directory of raw lyrics data",
    )
    parser.add_argument(
        "-v",
        "--val",
        type=float,
        default=0.2,
        help="Ratio of validation dataset",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="lyrcis_dataset",
        help="Path to output data",
    )

    args = parser.parse_args()

    return args


def mkdir_p(folder_path):
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)


def read_text(file_path):
    with open(file_path, "r") as f:
        return f.read()


def write_text(text, file_path):
    with open(file_path, "w") as f:
        f.write(text)


def transform_lyric(raw_lyric):
    lyric = HanziConv.toTraditional(raw_lyric)
    lyric = lyric.replace(" ", "，").replace("\n", "。")

    return lyric


def read_lyrics(data_root):
    lyrics = []

    singer_names = [
        f
        for f in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, f))
    ]
    for singer_name in singer_names:
        singer_dir = os.path.join(data_root, singer_name)
        song_names = [
            f
            for f in os.listdir(singer_dir)
            if os.path.isfile(os.path.join(singer_dir, f))
                and f.endswith(".txt")
        ]
        for song_name in song_names:
            raw_lyric = read_text(os.path.join(singer_dir, song_name))
            lyric = transform_lyric(raw_lyric)

            lyrics.append(lyric)

    return lyrics


def build_data(lyrics, output_path, bos_token="<BOS>", eos_token="<EOS>"):
    trans_lyrcis = []
    for lyric in lyrics:
        trans_lyric = str(lyric).strip()
        trans_lyric = re.sub(r"\s", " ", trans_lyric)
        trans_lyric = "{} {} {}".format(bos_token, trans_lyric, eos_token)

        trans_lyrcis.append(trans_lyric)

    data = "\n".join(trans_lyrcis)
    write_text(data, output_path)


def generate_dataset(lyrics, val_ratio, output_dir, random_state=1):
    lyrics_train, lyrics_val = train_test_split(
        lyrics,
        train_size=(1.0 - val_ratio),
        random_state=random_state,
    )

    build_data(lyrics_train, os.path.join(output_dir, "train.txt"))
    build_data(lyrics_val, os.path.join(output_dir, "val.txt"))


def main(args):
    mkdir_p(args.output)

    lyrics = read_lyrics(args.data)

    generate_dataset(lyrics, args.val, args.output)

    print("Generated dataset saved in {}".format(args.output))


if __name__ == "__main__":
    main(parse_args())
