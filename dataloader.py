import torchtext.data as data
import torchtext.vocab as vocab
import nltk
import torch

import numpy as np

from collections import Counter
import random
import itertools

# random.seed(1)

EOS_TOKEN = '<eos>'
SOS_TOKEN = '<sos>'
PAD_TOKEN = '<pad>'


class OpenSubtitle:
    """
    OpenSubtitle data parser
    """

    def __init__(self, filename, vocab_size=2000, batch_size=5, MAX_SENTENCE_LEN=10, shuffle=False):
        self.batch_size = batch_size
        self.specials = [PAD_TOKEN, EOS_TOKEN, SOS_TOKEN]
        self.shuffle = shuffle

        text = open(filename, mode='r', encoding='utf8').read().lower()
        words = text.split()
        self.words = Counter(words)

        self.vocab = self.words.most_common(vocab_size - len(self.specials))  # Returns list of tuples
        self.vocab = [word for word, count in self.vocab]
        self.vocab += self.specials

        # String to id : word2index
        # id to string : index2word
        self.index2word = dict(zip(range(len(self.vocab)), self.vocab))
        self.word2index = dict(zip(self.vocab, range(len(self.vocab))))

        self.vectors = []

        # Prepare the question, answer pair like in chat
        temp_q = text.splitlines()
        temp_q = [sentence.split() for sentence in temp_q]
        # Reject really long sentences, they're hard (slow) to train
        temp_q = [sentence for sentence in temp_q if len(sentence) < MAX_SENTENCE_LEN]

        # Filtering out sentences according to their lengths and OOV brings discontinuities in the dialog dataset
        # Reject sentences with Out of Vocabulary words in it
        self.q = []
        for dialog in temp_q:
            keep_dialog = True
            for word in dialog:
                if word not in self.word2index.keys():
                    keep_dialog = False
                    break
            if keep_dialog: self.q.append(dialog)

        print("Dialogues count preserved : {}/{}".format(len(self.q), len(temp_q)))
        del temp_q

        self.a = self.q[1:]
        self.q.pop()
        assert len(self.q) == len(self.a)

        self.num_batch = len(self.q) // self.batch_size

        self.generate_ids()

    def generate_ids(self):
        self.ids = list(range(len(self.q)))
        if self.shuffle:
            random.shuffle(self.ids)
        self.ids = self.ids[:self.num_batch * self.batch_size]

    def __next__(self):
        # Select list of ids
        ids = [self.ids.pop(0) for _ in range(self.batch_size)]

        # to start a new batch
        if not self.ids:
            self.generate_ids()

        # Return q, a belonging to those ids
        inputs = [self.word_id(self.q[id]) for id in ids]
        input_lengths = [len(dialog) for dialog in inputs]
        targets = [self.word_id(self.a[id]) for id in ids]

        # Sort the input according to decreasing length
        ranks = np.argsort(input_lengths)[::-1]
        input_lengths = torch.tensor(sorted(input_lengths, reverse=True))
        inputs = [inputs[id] for id in ranks]
        targets = [targets[id] for id in ranks]

        max_target_length = max([len(dialog) for dialog in targets])

        mask = [self.generate_mask(dialog, max_target_length) for dialog in targets]
        inputs = self.zeroPadding(inputs, self.word2index[PAD_TOKEN])
        targets = self.zeroPadding(targets, self.word2index[PAD_TOKEN])

        inputs = torch.tensor(inputs).view(-1, self.batch_size)
        targets = torch.tensor(targets).view(-1, self.batch_size)

        mask = torch.ByteTensor(mask).transpose(1, 0)

        return inputs, input_lengths, targets, mask, max_target_length

    def word_id(self, sentence):
        return [self.word2index[word] for word in sentence] + [self.word2index[EOS_TOKEN]]

    def generate_mask(self, dialog, max_length):
        return [1] * len(dialog) + [0] * (max_length - len(dialog))

    def zeroPadding(self, l, fillvalue):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def next(self):
        return self.__next__()

    def __len__(self):
        return self.num_batch


if __name__ == '__main__':
    filename = 'data/OpenSubtitles.en-eu.en'
    osp = OpenSubtitle(filename, 20000, 2, shuffle=True)

    data = osp.next()
    inputs, input_lengths, targets, mask, max_target_length = data

    print("Inputs : \n", inputs)
    print("Input l: ", input_lengths)
    print("Targets: \n", targets)
    print("Mask   : \n", mask)
    print("Max L  : ", max_target_length)

    # for j in range(3):
    #     print("New Batch started")
    #     for i in range(len(osp)):
    #         osp.next()
    #     print("Finished")

    # print(len(osp.vocab))

    # pretrained_embeddings = vocab.Vocab(osp.words, len(osp.vocab), 3, vectors="glove.6B.100d",
    #                                     vectors_cache='../.vector_cache').vectors
    # print(pretrained_embeddings.shape)
