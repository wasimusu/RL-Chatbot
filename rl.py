from __future__ import unicode_literals, print_function, division
import random

import numpy as np
import scipy.spatial

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

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


def RLStep(input_variable, target_variable, encoder, decoder, criterion, max_length=MAX_LENGTH,
           teacher_forcing_ratio=0.5, bidirectional=False):
    encoder_hidden = encoder.initHidden()

    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.to(device) if use_cuda else encoder_outputs

    loss = 0
    response = []

    # # TODO : This is to get attention
    # input_length = input_variable.size()[0]
    # # At the moment there is no attention thing going on
    # for ei in range(input_length):
    #     encoder_output, encoder_hidden = encoder(input_variable[ei], hidden=encoder_hidden)
    #     encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.to(device) if use_cuda else decoder_input

    if bidirectional:
        # sum the bidirectional hidden states into num_layers long cause the decoder is not bidirectional
        encoder_hidden = encoder_hidden[:encoder.num_layers, :, :] + encoder_hidden[encoder.num_layers:, :, :]

    decoder_hidden = encoder_hidden.to(device)  # TODO : Not the encoder's last hidden state

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # TODO : hacky patch
    target_variable = target_variable.to(device)

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            # print("Decoder output : ", decoder_input.size())
            # print(decoder_input)
            decoder_input = decoder_input.view(1, 1)  # The decoder expects 2d input
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):

            decoder_output, decoder_hidden = decoder(
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
            cos_sim = 1 - scipy.spatial.distance.cosine(vec_a.cpu(), vec_b.cpu())
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
        print("Input size : ", ep_input.size())
        print("Response size : ", curr_response.size())
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
        ep_rewards.append(rewards.item())

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


from dataloader import OpenSubtitle

filename = "data/OpenSubtitles.en-es.en"
filename = "data/OpenSubtitles.en-eu.en"
vocab_size = 50000
batch_size = 1


def trainRLIters(forward_encoder, forward_decoder, backward_encoder, backward_decoder, dull_responses, n_iters,
                 print_every=1000, plot_every=100, learning_rate=0.01, teacher_forcing_ratio=0.5,
                 bidirectional=False):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    osp = OpenSubtitle(filename, batch_size=batch_size, VOCAB_SIZE=vocab_size, shuffle=True)

    # Optimizer
    forward_encoder_optimizer = optim.SGD(forward_encoder.parameters(), lr=learning_rate)
    forward_decoder_optimizer = optim.SGD(forward_decoder.parameters(), lr=learning_rate)

    backward_encoder_optimizer = optim.SGD(backward_encoder.parameters(), lr=learning_rate)
    backward_decoder_optimizer = optim.SGD(backward_decoder.parameters(), lr=learning_rate)

    # training_pairs = [variablesFromPair(random.choice(pairs), reverse=False)
    #                   for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(len(osp)):
        training_batch = osp.next()
        # Extract fields from batch
        input_variable, input_lengths, target_variable, mask, max_target_len = training_batch

        # training_pair = training_pairs[iter - 1]
        # input_variable = training_pair[0]
        # target_variable = training_pair[1]

        ## Manually zero out the optimizer
        forward_encoder_optimizer.zero_grad()
        forward_decoder_optimizer.zero_grad()

        backward_encoder_optimizer.zero_grad()
        backward_decoder_optimizer.zero_grad()

        # Forward
        # print("Input size : ", input_variable.size())
        # print("Response size : ", target_variable.size())
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

        # print("Loss : ", loss.item(), loss)
        print_loss_total += (loss.cpu().item() / forward_len)
        plot_loss_total += (loss.cpu().item() / forward_len)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
