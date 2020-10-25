import numpy as np
import fasttext

from pure_fasttext import FastText

K = [1, 2, 3]
THRESHOLD = [0.0, 0.5]
DATA_LOC = "data/supervised.txt"
SENTENCE = [
    "this is text",
    "this is more text",
    "this   bananas text" "here is  text with    some   space  ",
    "   text's are fun with this is text" " this IS Text TEXT ",
]


def test_supervised_predict():
    model = fasttext.train_supervised(DATA_LOC)
    model_ = FastText(model)
    for k in K:
        for threshold in THRESHOLD:
            for sentence in SENTENCE:
                wv = model.predict(sentence, k=k, threshold=threshold)
                wv_ = model_.predict(sentence, k=k, threshold=threshold)
                assert wv[0] == wv_[0]
                assert np.allclose(wv[1], wv_[1], atol=1e-4)


def test_supervised_sentence():
    model = fasttext.train_supervised(DATA_LOC)
    model_ = FastText(model)
    for sentence in SENTENCE:
        wv = model.get_sentence_vector(sentence).tolist()
        wv_ = model_.get_sentence_vector(sentence)
        assert np.allclose(wv, wv_)
