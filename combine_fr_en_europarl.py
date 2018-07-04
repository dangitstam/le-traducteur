import os
import json
import sys

import argparse
from tqdm import tqdm


def main():
    """
    A pre-processing step before training Le Traducteur.
    
    Given the Europarl French and English parallel corpora, constructs a
    single corpus - a jsonl containing one example per line, where examples
    consist of a JSON object of the form:

    { id: Line #, "en": <The English Utterance>, "fr": <The French utterance> }

    Parallel corpora europarl-v7.fr-en.en,  europarl-v7.fr-en.fr are found here
    http://www.statmt.org/europarl/v7/fr-en.tgz
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))
    parser.add_argument("--english-path", type=str,
                        help="Path to the English Transcription from the"
                             "Europarl Corpus")
    parser.add_argument("--french-path", type=str,
                        help="Path to the French Transcription from the"
                             "Europarl Corpus")
    parser.add_argument("--save-dir", type=str,
                        default=project_root,
                        help="Directory to store the combined corpus.")
    args = parser.parse_args()

    out_path = os.path.join(args.save_dir, "europarl_en_fr.jsonl")
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
    print("Combining English and French Europarl Corpora:")
    with open(args.english_path, 'r') as en, open(args.french_path, 'r') as fr:
        for i, (en_utterance, fr_utterance) in tqdm(enumerate(zip(en, fr))):
            en_utterance = en_utterance.strip()
            fr_utterance = fr_utterance.strip()
            example = {
                'id': i,
                'en': en_utterance,
                'fr': fr_utterance
            }
            json.dump(example, out_file, ensure_ascii=False)
            out_file.write('\n')

    out_file.close()


if __name__ == "__main__":
    main()
