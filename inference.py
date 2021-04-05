import argparse

from transformers import AutoModelForCausalLM
from transformers import BertTokenizerFast
from transformers import TextGenerationPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="A script for lyrics generation (inference)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        type=str,
        help="Name of path to the model",
    )
    parser.add_argument(
        "tokenizer",
        type=str,
        help="Name of tokenizer",
    )

    args = parser.parse_args()

    return args


def unnormalize_lyric(lyric):
    return lyric.replace(" ", "").replace("，", " ").replace("。", "\n")


def generate_loop(lyrics_generator):
    while True:
        input_text = input("Input: ")

        outputs = lyrics_generator(input_text, max_length=1000, do_sample=True)
        output_text = outputs[0]["generated_text"]
        lyric = unnormalize_lyric(output_text)

        print("Output:")
        print(lyric)


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)

    lyrics_generator = TextGenerationPipeline(model, tokenizer)

    generate_loop(lyrics_generator)


if __name__ == "__main__":
    main(parse_args())
