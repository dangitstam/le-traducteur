import argparse
import json
import os
import shutil
import sys
import random

from tqdm import tqdm


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
    parser.add_argument("--train-percent", type=int,
                        default=80,
                        help="The percentage of the total corpus to reserve for training."
                             "Must be a whole number between 0 and 100")
    parser.add_argument("--dev-percent", type=int,
                        default=10,
                        help="The percentage of the total corpus to reserve for development"
                             "Must be a whole number between 0 and 100")
    parser.add_argument("--test-percent", type=int,
                        default=10,
                        help="The percentage of the total corpus to reserve for testing"
                             "Must be a whole number between 0 and 100")
    parser.add_argument("--seed", type=int,
                        default=1337,
                        help="The random seed in which the partition is made.")
    parser.add_argument("--save-dir", type=str,
                        default=project_root,
                        help="Directory to store the train-dev-test split directory.")
    args = parser.parse_args()

    # Enforce a valid train-dev-test split.
    if args.train_percent + args.dev_percent + args.test_percent != 100:
        raise ValueError("Provided partition percentages must add to 100.")

    train_ratio = args.train_percent / 100
    dev_ratio = args.dev_percent / 100
    test_ratio = args.test_percent / 100

    corpus_name = os.path.basename(args.corpus_path).split('.')[0]
    out_dir = os.path.join(args.save_dir, corpus_name)
    try:
        if os.path.exists(out_dir):
            input("Train-development-test split for corpus {} already exists.\n"
                  "Press <Ctrl-c> to exit or <Enter> to recreate it."
                  .format(args.corpus_path))
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
    except KeyboardInterrupt:
        print()
        sys.exit()

    train_file = open(os.path.join(out_dir, corpus_name + "_train.jsonl"), 'w')
    dev_file = open(os.path.join(out_dir, corpus_name + "_dev.jsonl"), 'w')
    test_file = open(os.path.join(out_dir, corpus_name + "_test.jsonl"), 'w')

    print("Reading dataset...")
    with open(args.corpus_path, 'r') as f:
        examples = f.readlines()

    num_all_examples = len(examples)
    random.seed(args.seed)
    random.shuffle(examples)

    num_train_examples = int(len(examples) * train_ratio)
    num_dev_examples = int(len(examples) * dev_ratio)
    num_test_examples = int(len(examples) * test_ratio)
    train_partition = examples[: num_train_examples]
    dev_partition = examples[num_train_examples: num_train_examples + num_dev_examples]
    test_partition = examples[num_dev_examples: num_dev_examples + num_test_examples]

    print("Creating train split:")
    for example in tqdm(train_partition):
        train_file.write(example)
    
    print("Creating dev split:")
    for example in tqdm(dev_partition):
        dev_file.write(example)
    
    print("Creating test split:")
    for example in tqdm(test_partition):
        test_file.write(example)

    # Log the way the dataset was created for reproducibility.
    partition_info = open(os.path.join(out_dir, "partition_info.txt"), 'w')
    partition_info.write(corpus_name.upper() + "\n")
    partition_info.write("-"*len(corpus_name) + "\n")
    partition_info.write("Random seed: {}\n".format(args.seed))
    partition_info.write("Total number of examples: {}\n".format(num_all_examples))
    partition_info.write("Training : Development : Test = {} : {} : {}\n"
                         .format(args.train_percent, args.dev_percent, args.test_percent))
    train_file.close()
    dev_file.close()
    test_file.close()
    partition_info.close()

if __name__ == "__main__":
    main()
