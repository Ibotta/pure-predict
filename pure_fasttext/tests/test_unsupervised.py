import numpy as np
import fasttext

from pure_fasttext import FastText

DATA_LOC = "data/unsupervised.txt"
SENTENCE = [
    "this is text",
    "this is more text",
    "this   bananas text" "here is  text with    some   space  ",
    "   text's are fun with this is text" " this IS Text TEXT ",
]


def test_unsupervised_word():
    model = fasttext.train_unsupervised(DATA_LOC, maxn=0)
    model_ = FastText(model)
    for sentence in SENTENCE:
        for word in sentence.split():
            wv = model.get_word_vector(word).tolist()
            wv_ = model_.get_word_vector(word)
            assert np.allclose(wv, wv_)


def test_unsupervised_sentence():
    model = fasttext.train_unsupervised(DATA_LOC, maxn=0)
    model_ = FastText(model)
    for sentence in SENTENCE:
        wv = model.get_sentence_vector(sentence).tolist()
        wv_ = model_.get_sentence_vector(sentence)
        assert np.allclose(wv, wv_)
