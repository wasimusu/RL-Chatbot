from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import scipy
import numpy as np
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)


# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i + 1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Initialize lines dict, conversations list, and field ids
lines = {}
conversations = []
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

# Load lines and process conversations
lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                  lines, MOVIE_CONVERSATIONS_FIELDS)

# Write new csv file
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)


MAX_LENGTH = 10  # Maximum sentence length to consider


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8'). \
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return 0 < len(p[0].split()) < MAX_LENGTH and 0 < len(p[1].split()) < MAX_LENGTH


# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    voc, pairs = readVocs(datafile, corpus_name)
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    for pair in pairs:
        p1 = pair[0].strip()
        p2 = pair[1].strip()
        if len(p1) < 2 or len(p2) < 2:
            continue
        voc.addSentence(p1)
        voc.addSentence(p2)
    print("Counted words:", voc.num_words)
    return voc, pairs


# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)

MIN_COUNT = 3  # Minimum word count threshold for trimming


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        # Check input sentence
        for word in input_sentence.split():
            if word not in voc.word2index:
                keep_input = False
                break

        # Check output sentence
        for word in output_sentence.split():
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs


# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split()] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def variableFromSentence(voc, sentence):
    indexes = indexesFromSentence(voc, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    result = result.to(device)
    return result


def variablesFromPair(pair, reverse):
    input_variable = variableFromSentence(voc, pair[0])
    target_variable = variableFromSentence(voc, pair[1])
    if reverse:
        return (target_variable, input_variable)
    return (input_variable, target_variable)


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split()), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])

    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, num_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          dropout=(0 if num_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths=[], hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)

        if isinstance(input_lengths, list):  # != None:
            input_lengths = [input_seq.size(0)]

        if not isinstance(input_lengths, list):  # != None:
            # Pack padded batch of sequences for RNN module
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        else:
            packed = embedded.unsqueeze(1)

        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)

        if not isinstance(input_lengths, list):  # != None:
            # Unpack padding
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # Return output and final hidden state
        return outputs, hidden


# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()
        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=(0 if num_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        encoder_outputs = encoder_outputs.transpose(0, 1)
        print("Attention weights , encoder_output: ", attn_weights.size(), encoder_outputs.size())

        # Modifications for RL training
        if encoder_outputs.dim() == 2:
            encoder_outputs = encoder_outputs.t()
            encoder_outputs = encoder_outputs.unsqueeze(1)

        context = attn_weights.bmm(encoder_outputs)
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        print(rnn_output.size(), context.size())
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding, dropout_p=0.1, num_layers=1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers

        self.embedding = embedding
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        hidden0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        self.hidden0 = nn.Parameter(hidden0, requires_grad=True)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return self.hidden0.to(device)


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.num_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding,
               encoder_num_layers, decoder_num_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip,
               corpus_name, loadFilename):
    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration - 1, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration,
                                                                                          iteration / n_iteration * 100,
                                                                                          print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name,
                                     '{}-{}_{}'.format(encoder_num_layers, decoder_num_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.num_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while (1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


# Configure models
model_name = 'cb_model'
attn_model = 'dot'
# attn_model = 'general'
# attn_model = 'concat'
hidden_size = 300
encoder_num_layers = 2
decoder_num_layers = encoder_num_layers
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 40000
loadFilename = os.path.join(save_dir, model_name, corpus_name,
                            '{}-{}_{}'.format(encoder_num_layers, decoder_num_layers, hidden_size),
                            '{}_checkpoint.tar'.format(checkpoint_iter))

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)

if loadFilename:
    embedding.load_state_dict(embedding_sd)
else:
    from torchtext import vocab
    from collections import Counter

    print("Initialized with pretrained embeddings")
    print("Vocab size : ", voc.num_words)

    counter = Counter(list(voc.index2word.values()))
    pretrained_embeddings = vocab.Vocab(counter, voc.num_words, vectors="glove.6B.300d",
                                        specials=[], vectors_cache='../.vector_cache').vectors
    print("Embedding size :", pretrained_embeddings.size())

    embedding.weight.data.copy_(pretrained_embeddings)

# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_num_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_num_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 40010
print_every = 50
save_every = 1000

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
# trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
#            embedding, encoder_num_layers, decoder_num_layers, save_dir, n_iteration, batch_size,
#            print_every, save_every, clip, corpus_name, loadFilename)

# Instantiate the backward encoder and decoder for RL
bidirectional = True
MIN_LENGTH = 2
backward_encoder = EncoderRNN(hidden_size, embedding, encoder_num_layers, dropout).to(device)
# backward_attn_decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_num_layers,
#                                             dropout).to(device)
backward_attn_decoder = AttnDecoderRNN(hidden_size, voc.num_words, embedding,
                                       dropout, decoder_num_layers).to(device)

print("Training backward Seq2Seq")

trainIters(model_name, voc, pairs, backward_encoder, backward_attn_decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_num_layers, decoder_num_layers, save_dir, 10000, batch_size,
           1, save_every, clip, corpus_name, loadFilename)

print("Finished training backward Seq2Seq")

forward_encoder = EncoderRNN(hidden_size, embedding, encoder_num_layers, dropout).to(device)
# forward_attn_decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_num_layers,
#                                            dropout).to(device)
forward_attn_decoder = AttnDecoderRNN(hidden_size, voc.num_words, embedding,
                                      dropout, decoder_num_layers).to(device)


def RLStep(input_variable, target_variable, encoder, decoder, criterion, max_length=MAX_LENGTH,
           teacher_forcing_ratio=0.5, bidirectional=False):
    encoder_hidden = None

    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.to(device)

    loss = 0
    response = []

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], hidden=encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.to(device)

    if bidirectional:
        # sum the bidirectional hidden states into num_layers long cause the decoder is not bidirectional
        encoder_hidden = encoder_hidden[:encoder.num_layers, :, :] + encoder_hidden[encoder.num_layers:, :, :]
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.to(device)

            print(target_variable[di].size(), decoder_output.size(), target_variable.size())
            loss += criterion(decoder_output, target_variable[di][0])

            # TODO: ni or decoder_output?
            response.append(ni)
            if ni == EOS_token:
                break

    return (loss, target_length, response)


######################################################################

def calculate_rewards(input_variable, target_variable,
                      forward_encoder, forward_decoder,
                      backward_encoder, backward_decoder,
                      criterion, dull_responses,
                      teacher_forcing_ratio, bidirectional):
    ep_rewards = []

    # ep_num are used to bound the number of episodes
    # MAXIMUM ep = 10
    ep_num = 1

    responses = []

    ep_input = input_variable
    ep_target = target_variable
    while (ep_num <= 10):

        # First start with Forward model to generate the current response, given ep_input
        # ep_target is empty if ep_num > 1
        _, _, curr_response = RLStep(ep_input, ep_target,
                                     forward_encoder, forward_decoder,
                                     criterion,
                                     teacher_forcing_ratio=teacher_forcing_ratio,
                                     bidirectional=bidirectional)

        ## Break once we see (1) dull response, (2) the response is less than MIN_LENGTH, (3) repetition
        if (len(curr_response) < MIN_LENGTH):  # or (curr_response in responses) or (curr_response in dull_responses):
            break

        curr_response = Variable(torch.LongTensor(curr_response), requires_grad=False).view(-1, 1)
        curr_response = curr_response.to(device)
        responses.append(curr_response)

        ## Ease of answering
        # Use the forward model to generate the log prob of generating dull response given ep_input.
        # Use the teacher_forcing_ratio = 1!
        r1 = 0
        for d in dull_responses:
            forward_loss, forward_len, _ = RLStep(ep_input, d, forward_encoder, forward_decoder,
                                                  criterion,
                                                  teacher_forcing_ratio=1.1,
                                                  bidirectional=bidirectional)
            if forward_len > 0:
                # log (1/P(a|s)) = CE  --> log(P(a | s)) = - CE
                r1 -= forward_loss / forward_len
        if len(dull_responses) > 0:
            r1 = r1 / len(dull_responses)

        ## Information flow
        # responses contains all the generated response by the forward model
        r2 = 0
        if (len(responses) > 2):
            # vec_a --> h_(i)  = responses[-3]
            # vec_b --> h_(i+1)= responses[-1]
            vec_a = responses[-3].data
            vec_b = responses[-1].data
            # length of the two vector might not match
            min_length = min(len(vec_a), len(vec_b))
            vec_a = vec_a[:min_length]
            vec_b = vec_b[:min_length]
            cos_sim = 1 - scipy.spatial.distance.cosine(vec_a, vec_b)
            # -1 <= cos_sim <= 1
            # TODO: how to handle negative cos_sim?
            if cos_sim <= 0:
                r2 = - cos_sim
            else:
                r2 = - np.log(cos_sim)

        ## Semantic Coherence
        # Use the forward model to generate the log prob of generating curr_response given ep_input
        # Use the backward model to generate the log prob of generating ep_input given curr_response
        r3 = 0
        forward_loss, forward_len, _ = RLStep(ep_input, curr_response,
                                              forward_encoder, forward_decoder,
                                              criterion,
                                              teacher_forcing_ratio=teacher_forcing_ratio,
                                              bidirectional=bidirectional)

        backward_loss, backward_len, _ = RLStep(curr_response, ep_input,
                                                backward_encoder, backward_decoder,
                                                criterion,
                                                teacher_forcing_ratio=teacher_forcing_ratio,
                                                bidirectional=bidirectional)
        if forward_len > 0:
            r3 += forward_loss / forward_len
        if backward_len > 0:
            r3 += backward_loss / backward_len

        ## Add up all the three rewards
        rewards = 0.25 * r1 + 0.25 * r2 + 0.5 * r3
        ep_rewards.append(rewards)

        ## Set the next input
        ep_input = curr_response
        ## TODO: what's the limit of the length? and what should we put as the dummy target?
        ep_target = Variable(torch.LongTensor([0] * MAX_LENGTH), requires_grad=False).view(-1, 1)
        ep_target = ep_target.to(device)

        # Turn off the teacher forcing ration after first iteration (since we don't have a target anymore).
        teacher_forcing_ratio = 0
        ep_num += 1

    # Take the mean of the episodic rewards
    r = 0

    if len(ep_rewards) > 0:
        r = np.mean(ep_rewards)

    return r


def trainRLIters(forward_encoder, forward_decoder, backward_encoder, backward_decoder, dull_responses, n_iters,
                 print_every=1000, plot_every=100, learning_rate=0.01, teacher_forcing_ratio=0.5,
                 bidirectional=False):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Optimizer
    forward_encoder_optimizer = optim.SGD(forward_encoder.parameters(), lr=learning_rate)
    forward_decoder_optimizer = optim.SGD(forward_decoder.parameters(), lr=learning_rate)

    backward_encoder_optimizer = optim.SGD(backward_encoder.parameters(), lr=learning_rate)
    backward_decoder_optimizer = optim.SGD(backward_decoder.parameters(), lr=learning_rate)

    training_pairs = []
    for _ in range(n_iteration):
        try:
            pair = random.choice(pairs)
            if len(pair) > 1:
                o = batch2TrainData(voc, pair)
                if len(o) == 5:
                    training_pairs.append(batch2TrainData(voc, pair))
        except Exception as e:
            pass

    print("Total number of pairs : ", len(training_pairs))

    # training_pairs_ = [batch2TrainData(voc, random.choice(pairs)) for i in range(n_iters)]
    # print("Total number of pairs 2 : ", len(training_pairs_))

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        input_variable, lengths, target_variable, mask, max_target_len = training_pairs[iter - 1]

        # training_pair = training_pairs[iter - 1]
        # input_variable = training_pair[0]
        # target_variable = training_pair[1]

        ## Manually zero out the optimizer
        forward_encoder_optimizer.zero_grad()
        forward_decoder_optimizer.zero_grad()

        backward_encoder_optimizer.zero_grad()
        backward_decoder_optimizer.zero_grad()

        # Forward
        forward_loss, forward_len, _ = RLStep(input_variable, target_variable,
                                              forward_encoder, forward_decoder,
                                              criterion,
                                              teacher_forcing_ratio=teacher_forcing_ratio,
                                              bidirectional=bidirectional)

        print("Did one RL Step. Now onto computing the reward")

        ## Calculate the reward
        reward = calculate_rewards(input_variable, target_variable,
                                   forward_encoder, forward_decoder,
                                   backward_encoder, backward_decoder,
                                   criterion, dull_responses,
                                   teacher_forcing_ratio, bidirectional)

        ## Update the forward seq2seq with its loss scaled by the reward
        loss = forward_loss * reward

        loss.backward()
        forward_encoder_optimizer.step()
        forward_decoder_optimizer.step()

        print_loss_total += (loss.data[0] / forward_len)
        plot_loss_total += (loss.data[0] / forward_len)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_loss_total = 0
            plot_losses.append(plot_loss_avg)


dull_responses = []

trainRLIters(forward_encoder, forward_attn_decoder, backward_encoder, backward_attn_decoder, dull_responses,
             n_iters=10000, print_every=100, learning_rate=0.0001,
             teacher_forcing_ratio=0.5, bidirectional=bidirectional)

# # Run Evaluation
# # ~~~~~~~~~~~~~~
# # Set dropout layers to eval mode
# encoder.eval()
# decoder.eval()
# # Initialize search module
# searcher = GreedySearchDecoder(encoder, decoder)
# # Begin chatting (uncomment and run the following line to begin)
# evaluateInput(encoder, decoder, searcher, voc)
