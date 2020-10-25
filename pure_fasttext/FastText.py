"""
Base class for fasttext API implemented in pure python
"""

from math import exp, sqrt
from operator import mul

EOS = "</s>"


def l2_norm(arr):
    """ Computes l2 norm of input list """
    return sqrt(sum(map(lambda x: x ** 2, arr)))


def div_norm(arr):
    """ Divides input list by l2 norm """
    norm_value = l2_norm(arr)
    if norm_value > 0:
        return list(map(lambda x: x * (1.0 / norm_value), arr))
    else:
        return arr


def dot(A, B):
    """
    Computes dot product between input 1-D list 'A'
    and 2-D list B.
    """
    arr = []
    for i in range(len(B)):
        val = sum(map(mul, A, B[i]))
        arr.append(val)
    return arr


def sfmax(arr):
    """ Computes softmax of input list """
    expons = list(map(lambda x: exp(x - max(arr)), arr))
    return list(map(lambda x: x / float(sum(expons)), expons))


def argsort(seq):
    """ Computes argsort of input list """
    return sorted(range(len(seq)), key=seq.__getitem__)


def average(l):
    """ Computes average of 2-D list """
    llen = len(l)

    def divide(x):
        return x / float(llen)

    return list(map(divide, map(sum, zip(*l))))


class _FastText(object):
    """
    Equivalent to fasttext `_FastText` class, implemented in
    pure python.
    """

    def __init__(self, model):
        self.supervised_ = model.label in model.get_labels()[0]
        if (model.maxn != 0) and not self.supervised_:
            raise ValueError(
                "Only maxn=0 is supported for unsupervised mode (no subwords support)."
            )
        if self.supervised_ and (model.loss.name not in ("softmax")):
            raise Exception("Only 'softmax' loss is supported for supervised mode.")
        self.words_ = model.get_words()
        self.input_matrix_ = model.get_input_matrix().tolist()
        self.dim_ = model.get_dimension()

        if self.supervised_:
            self.loss_ = model.loss.name
            self.labels_ = model.get_labels()
            self.output_matrix_ = model.get_output_matrix().tolist()
            self.is_quantized_ = model.is_quantized()

    def get_word_vector(self, word):
        """ Get the vector representation of word """
        idx = self.get_word_id(word)
        if idx >= 0:
            return self.get_input_vector(idx)
        else:
            return [0.0] * self.dim_

    def get_sentence_vector(self, text):
        """
        Given a string, get a single vector represenation. This function
        assumes to be given a single line of text. We split words on
        whitespace (space, newline, tab, vertical tab) and the control
        characters carriage return, formfeed and the null character.
        """
        if text.find("\n") != -1:
            raise ValueError("predict processes one line at a time (remove '\\n')")
        text += "\n"

        if self.supervised_:
            raw_words = [
                self.get_word_vector(word)
                for word in (text.split() + [EOS])
                if word in self.words_
            ]
            return average(raw_words)
        else:
            raw_words = [
                self.get_word_vector(word)
                for word in text.split()
                if word in self.words_
            ]
            return average(list(map(div_norm, raw_words)))

    def get_word_id(self, word):
        """
        Given a word, get the word id within the dictionary.
        Returns -1 if word is not in the dictionary.
        """
        if word in self.words_:
            return self.words_.index(word)
        else:
            return -1

    def get_input_vector(self, ind):
        """
        Given an index, get the corresponding vector of the Input Matrix.
        """
        return self.input_matrix_[ind]

    def _predict(self, text, k=1, threshold=0.0):
        """ Guts of prediction method """
        A = self.get_sentence_vector(text)
        if self.loss_ == "softmax":
            preds = sfmax(dot(A, self.output_matrix_))
        else:
            raise ValueError("Predict is not supported for loss: {}".format(self.loss_))
        argsorted = [i for i in argsort(preds)[::-1] if preds[i] >= threshold]
        pred_labels = tuple([self.labels_[i] for i in argsorted[:k]])
        pred_scores = [preds[i] for i in argsorted[:k]]
        return (pred_labels, pred_scores)

    def predict(self, text, k=1, threshold=0.0):
        """
        Given a string, get a list of labels and a list of
        corresponding probabilities.
        """
        if not self.supervised_:
            raise ValueError("Model needs to be supervised for prediction!")

        if isinstance(text, list):
            text = [entry for entry in text]
            all_labels = []
            all_probs = []
            all_labels, all_probs = self.f.multilinePredict(
                text, k, threshold, on_unicode_error
            )
            for text_ in text:
                labels, probs = self._predict(text_, k, threshold)
                all_labels.append(labels)
                all_probs.append(probs)
            return all_labels, all_probs
        else:
            return self._predict(text, k, threshold)
