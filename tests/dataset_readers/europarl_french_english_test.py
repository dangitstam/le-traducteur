

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from library.dataset_readers.europarl_french_english import EuroparlDatasetReader

class TestEuroparlDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = EuroparlDatasetReader()
        dataset = reader.read('tests/fixtures/smoke_europarl_en_fr.jsonl')
        instances = ensure_list(dataset)
        instance0 = {
            "id": 0,
            "en": ["Resumption", "of", "the", "session"],
            "fr": ["Reprise", "de", "la", "session"]
        }
        instance1 = {
            "id": 1,
            "en": ["I", "declare", "resumed", "the", "session", "of", "the", "European",
                   "Parliament", "adjourned", "on", "Friday", "17", "December", "1999",
                   ",", "and", "I", "would", "like", "once", "again", "to", "wish",
                   "you", "a", "happy", "new", "year", "in", "the", "hope", "that",
                   "you", "enjoyed", "a", "pleasant", "festive", "period", "."],
            "fr": ["Je", "déclare", "reprise", "la", "session", "du", "Parlement",
                   "européen", "qui", "avait", "été", "interrompue", "le", "vendredi",
                   "17", "décembre", "dernier", "et", "je", "vous", "renouvelle", "tous",
                   "mes", "vux", "en", "espérant", "que", "vous", "avez", "passé", "de",
                   "bonnes", "vacances", "." ]
        }
        instance7 = {
            "id": 7, 
            "en": ["Madam", "President", ",", "on", "a", "point", "of", "order", "."],
            "fr": ["Madame", "la", "Présidente", ",", "c'", "est", "une", "motion", "de",
                   "procédure", "."]
        }

        assert len(instances) == 10
        fields = instances[0].fields
        assert [t.text for t in fields["en"].tokens] == instance0["en"]
        assert [t.text for t in fields["fr"].tokens] == instance0["fr"]
        fields = instances[1].fields
        assert [t.text for t in fields["en"].tokens] == instance1["en"]
        assert [t.text for t in fields["fr"].tokens] == instance1["fr"]
        fields = instances[7].fields
        assert [t.text for t in fields["en"].tokens] == instance7["en"]
        assert [t.text for t in fields["fr"].tokens] == instance7["fr"]
