import unittest
from src.data_tools import *
from src.text_tools import recursive_remove_unks



class DataTester(unittest.TestCase):
    def test_get_data(self):
        sentences, sentences_encoded, char_dict, char_dict_inverse = get_sentences("TATOEBA", max_length=64)
        codes_batch_gen = batching(list_of_iterables=[sentences_encoded, sentences],
                                   n=512,
                                   infinite=False,
                                   return_incomplete_batches=False)

        for sentences_encoded, sentences in codes_batch_gen:
            assert len(sentences_encoded) == len(sentences) == 512
            sentences_decoded = ["".join(list(map(char_dict_inverse.get, s))) for s in sentences_encoded]
            already_seen = set()
            for s_original, s_codec in zip(sentences, sentences_decoded):
                s_codec = recursive_remove_unks(s_codec) # remove padding
                self.assertTrue(s_codec.startswith("<START>"))  # Check all of them start with the proper symbol
                self.assertTrue(s_codec.endswith("<END>"))  # Check all of them finish with the proper symbol
                self.assertEqual(s_original, s_codec[7:-5])  # Check if the decoded sentences are correct
                self.assertNotIn(s_original, already_seen)  # Check there are no duplicates in the data
                already_seen.add(hash(s_original))


