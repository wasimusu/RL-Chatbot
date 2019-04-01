import torchtext.data as data
import torchtext.vocab as vocab
import nltk

from collections import Counter
import random


class OpenSubtitle:
    """"""
    def __init__(self, filename, MIN_FREQ=1, VOCAB_SIZE=2000, batch_size=2, MAX_SENTENCE_LEN=20, shuffle=True):
        self.batch_size = batch_size
        self.specials = ['<pad>', '<eos>', '<sos>']
        self.shuffle = shuffle

        text = open(filename, mode='r', encoding='utf8').read().lower()
        words = text.split()
        self.words = Counter(words)

        self.vocab = self.words.most_common(VOCAB_SIZE)
        self.vocab += self.specials

        # String to id : stoi
        # id to string : itos
        self.itos = dict(zip(range(len(self.vocab)), self.vocab))
        self.stoi = dict(zip(range(len(self.vocab)), self.vocab))

        self.vectors = []

        # Prepare the question, answer pair like in chat
        self.q = text.splitlines()
        self.q = [sentence.split() for sentence in self.q]
        # Reject really long sentences, they're hard (slow) to train
        self.q = [sentence for sentence in self.q if len(sentence) < MAX_SENTENCE_LEN]
        self.a = self.q[1:]
        self.q.pop()
        assert len(self.q) == len(self.a)

        self.ids = list(range(len(self.q)))  # Need to re-instantiate this for another epoch of training

    def __next__(self):
        # Select list of ids
        if self.shuffle:
            ids = random.choices(self.ids, k=self.batch_size)
        else:
            ids = [self.ids.pop(0) for _ in range(self.batch_size)]

        # Now remove those id from ids so that they are not sampled again
        for id in ids:
            self.ids.pop(id)

        # Return q, a belonging to those ids
        q = [self.q[id] for id in ids]
        a = [self.a[id] for id in ids]
        return q, a

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.q)


if __name__ == '__main__':
    osp = OpenSubtitle(filename='data/opensubtitle/en-eu.txt/OpenSubtitles.en-eu.en', MIN_FREQ=3)
    data = osp.next()
    # for q, a in data:
    #     print(q, a)
    vocab = vocab.Vocab(osp.words, 2000, 3, specials=['<eos>', '<sos>', '<pad>'], vectors="glove.6B.100d",
                        vectors_cache='../.vector_cache')
    print(vocab.vectors[4])
