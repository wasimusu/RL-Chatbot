from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle

import numpy as np
import scipy.spatial

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = ('cuda' if use_cuda else 'cpu')

print('Device : ', device)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Maximum length of the sequences you are mapping
MAX_LENGTH = 20  # 50
MIN_LENGTH = 4  # 2

# Dull set (from RL-chatbot)
dull_set = ["I don't know what you're talking about.", "I don't know.",
            "You don't know.", "You know what I mean.", "I know what you mean.",
            "You know what I'm saying.", "You don't know anything."]

# start of sentence and end of sentence indices
SOS_token = 0
EOS_token = 1

input_lang = pickle.load(open("saved_pickle/input_lang_4_20.p", "rb"))
output_lang = pickle.load(open("saved_pickle/output_lang_4_20.p", "rb"))
pairs = pickle.load(open("saved_pickle/pairs_4_20.p", "rb"))

# input_lang.n_words, input_lang.word2index["froid"], input_lang.index2word[33]
input_lang.n_words, output_lang.n_words, output_lang.word2index["?"]


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub("newlinechar", "", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('../data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    '''
    Your Preferences here
    '''
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and len(
        p[1].split(' ')) > MIN_LENGTH and "https://" not in p[1]


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False, Filter=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    if Filter:
        pairs = filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('input', 'output', reverse=False, Filter=True)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.to(device)
    else:
        return result


def variablesFromPair(pair, reverse):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    if reverse:
        return (target_variable, input_variable)
    return (input_variable, target_variable)


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


import numpy as np


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=3, bidirectional=False):

        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, bidirectional=bidirectional)

        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        # make the initial hidden state learnable as well 
        hidden0 = torch.zeros(self.num_layers * num_directions, 1, self.hidden_size)

        if use_cuda:
            hidden0 = hidden0.to(device)
        else:
            hidden0 = hidden0

        self.hidden0 = nn.Parameter(hidden0, requires_grad=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)

        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # Sum bidirectional outputs

        return output, hidden

    def initHidden(self):

        if use_cuda:
            return self.hidden0.to(device)
        else:
            return self.hidden0


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH, num_layers=3):

        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        hidden0 = torch.zeros(self.num_layers, 1, self.hidden_size)

        if use_cuda:
            hidden0 = hidden0.to(device)
        else:
            hidden0 = hidden0

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

        if use_cuda:
            return self.hidden0.to(device)
        else:
            return self.hidden0


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH, teacher_forcing_ratio=0.5, bidirectional=False):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.to(device) if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)

        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.to(device) if use_cuda else decoder_input

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
            decoder_input = decoder_input.to(device) if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100,
               learning_rate=0.01, teacher_forcing_ratio=0.5, bidirectional=False, reverse=False):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [variablesFromPair(random.choice(pairs), reverse)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion,
                     teacher_forcing_ratio=teacher_forcing_ratio,
                     bidirectional=bidirectional)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


hidden_size = 128  # 256
num_layers = 2  # 4
bidirectional = False  # True
encoder = EncoderRNN(input_lang.n_words, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, num_layers=num_layers)

if use_cuda:
    encoder = encoder.to(device)
    attn_decoder = attn_decoder.to(device)

# encoder.load_state_dict(torch.load("saved_params/encoder.pth"))
# attn_decoder.load_state_dict(torch.load("saved_params/attn_decoder.pth"))
# encoder.load_state_dict(torch.load("saved_params/encoder_2L_h128_uni.pth"))
# attn_decoder.load_state_dict(torch.load("saved_params/attn_decoder_2L_h128_uni.pth"))

trainIters(encoder, attn_decoder, n_iters=10000, print_every=1000,
           learning_rate=0.0001, teacher_forcing_ratio=0.75,
           bidirectional=bidirectional)  # last loss 4.3393, 16 secs per 100 iters, so ~ 22500 iters/hr

# If you want to save the results of your training
torch.save(encoder.state_dict(), "saved_params/encoder_2L_h128_uni.pth")
torch.save(attn_decoder.state_dict(), "saved_params/attn_decoder_2L_h128_uni.pth")


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH,
             bidirectional=bidirectional):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.to(device) if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.to(device) if use_cuda else decoder_input

    if bidirectional:
        # sum the bidirectional hidden states into num_layers long cause the decoder is not bidirectional
        encoder_hidden = encoder_hidden[:encoder.num_layers, :, :] + encoder_hidden[encoder.num_layers:, :, :]

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.to(device) if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10, bidirectional=False):
    for i in range(n):
        pair = random.choice(pairs)
        print('input from data >', pair[0])
        print('output from data=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], bidirectional=bidirectional)
        output_sentence = ' '.join(output_words)
        print('bot response <', output_sentence)
        print('')


evaluateRandomly(encoder, attn_decoder, n=10, bidirectional=bidirectional)

bidirectional = False  # True

hidden_size = 128  # 256
num_layers = 2  # 4
bidirectional = False  # True
backward_encoder = EncoderRNN(output_lang.n_words, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
backward_attn_decoder = AttnDecoderRNN(hidden_size, input_lang.n_words, dropout_p=0.1, num_layers=num_layers)

if use_cuda:
    backward_encoder = backward_encoder.to(device)
    backward_attn_decoder = backward_attn_decoder.to(device)

# backward_encoder.load_state_dict(torch.load("saved_params/backward_encoder_2L_h128_uni.pth"))
# backward_attn_decoder.load_state_dict(torch.load("saved_params/backward_attn_decoder_2L_h128_uni.pth"))


trainIters(backward_encoder, backward_attn_decoder, n_iters=10000, print_every=1000,
           learning_rate=0.0001, teacher_forcing_ratio=0.75,
           bidirectional=bidirectional, reverse=True)

# If you want to save the results of your training
torch.save(backward_encoder.state_dict(), "saved_params/backward_encoder_2L_h128_uni.pth")
torch.save(backward_attn_decoder.state_dict(), "saved_params/backward_attn_decoder_2L_h128_uni.pth")

hidden_size = 128  # 256
num_layers = 2  # 4
bidirectional = False  # True
# Forward
forward_encoder = EncoderRNN(input_lang.n_words, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
forward_attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, num_layers=num_layers)

# Backward
backward_encoder = EncoderRNN(output_lang.n_words, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
backward_attn_decoder = AttnDecoderRNN(hidden_size, input_lang.n_words, dropout_p=0.1, num_layers=num_layers)

if use_cuda:
    forward_encoder = forward_encoder.to(device)
    forward_attn_decoder = forward_attn_decoder.to(device)

    backward_encoder = backward_encoder.to(device)
    backward_attn_decoder = backward_attn_decoder.to(device)


# Forward
# forward_encoder.load_state_dict(torch.load("saved_params/encoder_2L_h128_uni.pth"))
# forward_attn_decoder.load_state_dict(torch.load("saved_params/attn_decoder_2L_h128_uni.pth"))

# backward_encoder.load_state_dict(torch.load("saved_params/backward_encoder_2L_h128_uni.pth"))
# backward_attn_decoder.load_state_dict(torch.load("saved_params/backward_attn_decoder_2L_h128_uni.pth"))


def RLStep(input_variable, target_variable, encoder, decoder, criterion, max_length=MAX_LENGTH,
           teacher_forcing_ratio=0.5, bidirectional=False):
    encoder_hidden = encoder.initHidden()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.to(device) if use_cuda else encoder_outputs

    loss = 0
    response = []

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)

        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.to(device) if use_cuda else decoder_input

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
            decoder_input = decoder_input.to(device) if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])

            # TODO: ni or decoder_output?
            response.append(ni)
            if ni == EOS_token:
                break

    return (loss, target_length, response)


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
        curr_response = curr_response.to(device) if use_cuda else curr_response
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
        ep_target = ep_target.to(device) if use_cuda else ep_target

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
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Optimizer
    forward_encoder_optimizer = optim.SGD(forward_encoder.parameters(), lr=learning_rate)
    forward_decoder_optimizer = optim.SGD(forward_decoder.parameters(), lr=learning_rate)

    backward_encoder_optimizer = optim.SGD(backward_encoder.parameters(), lr=learning_rate)
    backward_decoder_optimizer = optim.SGD(backward_decoder.parameters(), lr=learning_rate)

    training_pairs = [variablesFromPair(random.choice(pairs), reverse=False)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

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
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


dull_responses = [variableFromSentence(output_lang, d) for d in dull_set]
dull_responses = []

trainRLIters(forward_encoder, forward_attn_decoder, backward_encoder, backward_attn_decoder, dull_responses,
             n_iters=10000, print_every=100, learning_rate=0.0001,
             teacher_forcing_ratio=0.5, bidirectional=bidirectional)

torch.save(forward_encoder.state_dict(), "saved_params/rl_forward_encoder_2L_h128_uni.pth")
torch.save(forward_attn_decoder.state_dict(), "saved_params/rl_forward_attn_decoder_2L_h128_uni.pth")

evaluateRandomly(forward_encoder, forward_attn_decoder, n=10, bidirectional=bidirectional)
