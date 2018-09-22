import argparse
import os
import random
import shutil
import sys
from itertools import zip_longest

from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from tqdm import tqdm

import ujson


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def main():
    """
    Given a parallel corpus, partitions examples into training, development, and test sets.

    Provided output will be a directory containing the partitions:
    <corpus_name> /
        <corpus_name>_train.jsonl
        <corpus_name>_development.jsonl
        <corpus_name>_test.jsonl
        partition_info.txt

    when given a parallel corpus <corpus_name>.jsonl
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(  # Escape out into project directory.
            os.path.dirname( # Escape out into scripts directory.
                os.path.realpath(__file__))))))
    parser.add_argument("--corpus-path", type=str,
                        help="Path to the parallel JSON lines corpus.")
    parser.add_argument("--save-dir", type=str,
                        default=project_root,
                        help="Directory to store the train-dev-test split directory.")
    args = parser.parse_args()
    corpus_name = os.path.basename(args.corpus_path).split('.')[0]
    out_file_path = os.path.join(args.save_dir, corpus_name + "_tokenized.jsonl")
    out_file = open(out_file_path, 'w')

    # Language-specific tokenizers.
    en_tokenizer = SpacyWordSplitter(language='en_core_web_sm')
    fr_tokenizer = SpacyWordSplitter(language='fr_core_news_sm')

    print("Tokenizing utterances for {}...".format(corpus_name))
    with open(args.corpus_path) as f:
        for lines in tqdm(grouper(f, 100, '')):
            # When the grouper collects a group smaller than the batch, padding
            # is done via empty strings.
            # Check for them explicitly before continuing.
            examples = [ujson.loads(line.strip()) for line in filter(lambda l: l, lines)]
            en_utterances = [ex['en'] for ex in examples]
            fr_utterances = [ex['fr'] for ex in examples]

            en_utterances_tokenized = en_tokenizer.batch_split_words(en_utterances)
            fr_utterances_tokenized = fr_tokenizer.batch_split_words(fr_utterances)

            for i, ex in enumerate(examples):
                ex_tokenized = {
                    'id': ex['id'],
                    'en': ' '.join([token.text for token in en_utterances_tokenized[i]]),
                    'fr': ' '.join([token.text for token in fr_utterances_tokenized[i]])
                }
                ujson.dump(ex_tokenized, out_file, ensure_ascii=False)
                out_file.write('\n')

    out_file.close()


if __name__ == "__main__":
    main()
