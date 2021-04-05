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
    parser.add_argument(
        "--max_length",
        type=int,
        default=1000,
        help="Maximum length for generation",
    )
    parser.add_argument(
        "--dont_sample",
        action="store_true",
        help="Do not use sampling",
    )

    args = parser.parse_args()

    return args


def unnormalize_lyric(lyric):
    return lyric.replace(" ", "").replace("，", " ").replace("。", "\n")


def generate_loop(lyrics_generator, args):
    while True:
        input_text = input("Input: ")

        outputs = lyrics_generator(
            [input_text],
            max_length=args.max_length,
            do_sample=not args.dont_sample,
        )
        output_text = outputs[0]["generated_text"]
        lyric = unnormalize_lyric(output_text)

        print("Output:")
        print(lyric)


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)

    lyrics_generator = TextGenerationPipeline(model, tokenizer)

    generate_loop(lyrics_generator, args)


if __name__ == "__main__":
    main(parse_args())
