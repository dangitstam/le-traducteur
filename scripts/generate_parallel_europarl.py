import argparse
import json
import os
import sys

from tqdm import tqdm


def main():
    """
    A pre-processing step before training Le Traducteur.
    
    Given files from the Europarl parallel corpora, constructs a single corpus - a jsonl containing
    one example per line, where examples consist of a JSON object of the form:
    {
      "id": Line #,
      <src-language>: <the src utterance>,
      <dst-language>: <the dst utterance>
    }

    Parallel corpora are found here - http://www.statmt.org/europarl/
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))
    parser.add_argument("--src-path", type=str,
                        help="Path to the source language transcription from the"
                             "Europarl Corpus.")
    parser.add_argument("--dst-path", type=str,
                        help="Path to the destination language transcription from the"
                             "Europarl Corpus.")
    parser.add_argument("--src-language", type=str,
                        help="Name of the source language (ISO code).")
    parser.add_argument("--dst-language", type=str,
                        help="Name of the destination language (ISO code).")
    parser.add_argument("--save-dir", type=str,
                        default=project_root,
                        help="Directory to store the combined corpus.")
    args = parser.parse_args()

    out_path = os.path.join(args.save_dir, "europarl_{}_{}.jsonl"
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
    print("Combining Transcriptions from Europarl Corpora into a Parallel Set:")
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
