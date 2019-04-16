"""
Contains all the knobs for the whole project
"""
import torch


class Config:
    def __init__(self):
        # RNN params
        self.embedding_size = 50
        self.hidden_size = self.embedding_size
        self.batch_size = 16

        self.max_sentence_len = 10
        self.vocab_size = 10000

        self.corpus_name = "OpenSubtitles"
        self.filename = "data/OpenSubtitles.en-es.en"
        self.filename = "data/OpenSubtitles.en-eu.en"

        # optimizer params
        self.learning_rate = 0.01
        self.l2_norm = 0.1
        self.num_epochs = 10

        # directories
        self.data_dir = "data/"
        self.vectors_cache = "../vectors_cache"
        self.save_dir = "model/"

        # Logs
        self.print_every = 100
        self.save_every = 1  # Save every 10000 iterations

        self.use_cuda = torch.cuda.is_available()
        self.device = ('cuda' if self.use_cuda else 'cpu')
        self.mode = "train"
        # self.mode = 'inference'
        self.teacher_forcing_ratio = 0.9 if self.mode == 'train' else 0
