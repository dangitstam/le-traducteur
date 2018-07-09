import argparse
import json
import os
import sys

from tqdm import tqdm


def main():
    """
    A tool for producing a parallel corpus with which to train Le Traducteur.
    
    Given a pair of monolingual transcriptions where transcriptions differ in language, constructs
    a single corpus - a jsonl containing one example per line.
    
    Each example consists of a JSON object of the form:
    {
      "id": Line #,
      <src-language>: <the src utterance>,
      <dst-language>: <the dst utterance>
    }
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(  # Escape out into project directory.
            os.path.dirname( # Escape out into scripts directory.
                os.path.realpath(__file__))))))
    parser.add_argument("--corpus-name", type=str,
                        default=None,
                        help="Optional: The name of the corpus from which the transcriptions"
                             "were found. If provided, this value will be pre-pended to the name"
                             "of the final jsonl.")
    parser.add_argument("--src-path", type=str,
                        help="Path to the source language transcription.")
    parser.add_argument("--dst-path", type=str,
                        help="Path to the destination language transcription.")
    parser.add_argument("--src-language", type=str,
                        help="Name of the source language (ISO code).")
    parser.add_argument("--dst-language", type=str,
                        help="Name of the destination language (ISO code).")
    parser.add_argument("--save-dir", type=str,
                        default=project_root,
                        help="Directory to store the combined corpus.")
    args = parser.parse_args()

    if args.corpus_name:
        out_path = os.path.join(args.save_dir, "{}_parallel_{}_{}.jsonl"
                                            .format(args.corpus_name, args.src_language,
                                                    args.dst_language))
    else:
        out_path = os.path.join(args.save_dir, "parallel_{}_{}.jsonl"
                                               .format(args.src_language, args.dst_language))
    try:
        if os.path.exists(out_path):
            input("Combined corpus {} already exists.\n"
                  "Press <Ctrl-c> to exit or "
                  "<Enter> to recreate it.".format(out_path))
    except KeyboardInterrupt:
        print()
        sys.exit()

    out_file = open(out_path, 'w')

    # Processes the two transcriptions in parallel.
    print("Combining monolingual transcipts into a parallel corpus:")
    with open(args.src_path, 'r') as en, open(args.dst_path, 'r') as fr:
        for i, (src_utterance, dst_utterance) in tqdm(enumerate(zip(en, fr))):
            src_utterance = src_utterance.strip()
            dst_utterance = dst_utterance.strip()
            example = {
                'id': i,
                args.src_language: src_utterance,
                args.dst_language: dst_utterance
            }
            json.dump(example, out_file, ensure_ascii=False)
            out_file.write('\n')

    out_file.close()


if __name__ == "__main__":
    main()
