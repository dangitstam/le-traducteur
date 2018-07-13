from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from library.dataset_readers.europarl_english_french import (EuroparlEnglishFrenchReader,
                                                             EuroparlEnglishFrenchReaderPretokenized)


class TestEuroparlEnglishFrenchReader(AllenNlpTestCase):
    # Hard-coded excerpts from the corpus for smoke testing.
    INSTANCE_0 = {
        "id": 0,
        "en": ["Resumption", "of", "the", "session"],
        "fr": ["Reprise", "de", "la", "session"]
    }
    INSTANCE_1 = {
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
    INSTANCE_7 = {
        "id": 7, 
        "en": ["Madam", "President", ",", "on", "a", "point", "of", "order", "."],
        "fr": ["Madame", "la", "Présidente", ",", "c'", "est", "une", "motion", "de",
                "procédure", "."]
    }

    DATASET_PATH = 'tests/fixtures/smoke_europarl_en_fr.jsonl'
    PRETOKENIZED_DATASET_PATH = 'tests/fixtures/smoke_europarl_en_fr_tokenized.jsonl'

    def test_read_from_file(self):
        reader = EuroparlEnglishFrenchReader()
        dataset = reader.read(TestEuroparlEnglishFrenchReader.DATASET_PATH)
        instances = ensure_list(dataset)

        assert len(instances) == 10
        fields = instances[0].fields
        assert [t.text for t in fields["en"].tokens] == TestEuroparlEnglishFrenchReader.INSTANCE_0["en"]
        assert [t.text for t in fields["fr"].tokens] == TestEuroparlEnglishFrenchReader.INSTANCE_0["fr"]
        fields = instances[1].fields
        assert [t.text for t in fields["en"].tokens] == TestEuroparlEnglishFrenchReader.INSTANCE_1["en"]
        assert [t.text for t in fields["fr"].tokens] == TestEuroparlEnglishFrenchReader.INSTANCE_1["fr"]
        fields = instances[7].fields
        assert [t.text for t in fields["en"].tokens] == TestEuroparlEnglishFrenchReader.INSTANCE_7["en"]
        assert [t.text for t in fields["fr"].tokens] == TestEuroparlEnglishFrenchReader.INSTANCE_7["fr"]


    def test_read_from_file_pretokenized(self):
        reader = EuroparlEnglishFrenchReaderPretokenized()
        dataset = reader.read(TestEuroparlEnglishFrenchReader.PRETOKENIZED_DATASET_PATH)
        instances = ensure_list(dataset)

        assert len(instances) == 10
        fields = instances[0].fields
        assert [t.text for t in fields["en"].tokens] == TestEuroparlEnglishFrenchReader.INSTANCE_0["en"]
        assert [t.text for t in fields["fr"].tokens] == TestEuroparlEnglishFrenchReader.INSTANCE_0["fr"]
        fields = instances[1].fields
        assert [t.text for t in fields["en"].tokens] == TestEuroparlEnglishFrenchReader.INSTANCE_1["en"]
        assert [t.text for t in fields["fr"].tokens] == TestEuroparlEnglishFrenchReader.INSTANCE_1["fr"]
        fields = instances[7].fields
        assert [t.text for t in fields["en"].tokens] == TestEuroparlEnglishFrenchReader.INSTANCE_7["en"]
        assert [t.text for t in fields["fr"].tokens] == TestEuroparlEnglishFrenchReader.INSTANCE_7["fr"]
