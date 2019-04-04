from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import os

from torch import optim
from torchtext import vocab

from dataloader import OpenSubtitle
from network import *

save_dir = 'model/'
MAX_LENGTH = 10  # Maximum sentence length to consider
corpus_name = 'opensubtitles'
batch_size = 128

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.1
learning_rate = 0.0005
decoder_learning_ratio = 5.0
print_every = 100
save_every = 1000  # Save every 10000 iterations

# Configure models
attn_model = 'dot'
# attn_model = 'general'
# attn_model = 'concat'
hidden_size = 300
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1

# Set checkpoint to load from; set to None if starting from scratch
dir = os.path.join(save_dir, corpus_name,
                   '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
if not os.path.exists(dir): os.makedirs(dir)
checkpoints = os.listdir(dir)
checkpoints = [int(filename.split('_')[0]) for filename in checkpoints]
if checkpoints == []:
    checkpoints = 1
else:
    checkpoints = max(checkpoints)
loadFilename = os.path.join(dir, '{}_checkpoint.tar'.format(checkpoints))
print(loadFilename, os.path.exists(loadFilename))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filename = "data/OpenSubtitles.en-es.en"
filename = "data/OpenSubtitles.en-eu.en"
osp = OpenSubtitle(filename, batch_size=batch_size, VOCAB_SIZE=50000, shuffle=True)
data = osp.next()
inputs, input_lengths, targets, mask, max_target_length = data

SOS_TOKEN = osp.word2index['<sos>']
EOS_TOKEN = osp.word2index['<eos>']
PAD_TOKEN = osp.word2index['<pad>']
vocab_size = len(osp.vocab)
print("Vocab size : ", vocab_size)


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, encoder_optimizer,
          decoder_optimizer, clip):
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

    # print("Encoder outputs : {} Hidden : {}\nMaybe only includes the last hidden state and output \n".format(
    #     encoder_outputs.size(), encoder_hidden.size()))

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_TOKEN for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    # print("Here we are using only last state of decoder for encoder's hidden state : ", decoder_hidden.size())

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


def trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_n_layers, decoder_n_layers,
               clip, loadFilename):
    # Initializations
    print_loss = 0

    epoch = 1
    if loadFilename and os.path.exists(loadFilename):
        epoch = checkpoint['epoch']

    # Training loop
    num_epochs = 20
    print("Training...")
    for epoch in range(epoch, num_epochs + 1):
        epoch_loss = 0
        for iteration in range(osp.__len__() - 2):
            # Load batches for each iteration
            training_batch = osp.next()
            # Extract fields from batch
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            # Run a training iteration with batch
            loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
                         encoder_optimizer, decoder_optimizer, clip)
            print_loss += loss
            epoch_loss += loss

            # Print progress
            if iteration % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("Epoch: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(epoch,
                                                                                          iteration / len(
                                                                                              osp) * 100,
                                                                                          print_loss_avg))
                print_loss = 0

            # Save checkpoint
            if (iteration % save_every == 0):
                directory = os.path.join(save_dir, corpus_name,
                                         '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'epoch': epoch,
                    'en': encoder.state_dict(),
                    'de': decoder.state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'de_opt': decoder_optimizer.state_dict(),
                    'loss': loss,
                    'embedding': embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))


class GreedySearchDecoder(nn.Module):
    """
    Computation Graph

       1) Forward input through encoder model.
       2) Prepare encoder's final hidden layer to be first hidden input to the decoder.
       3) Initialize decoder's first input as SOS_TOKEN .
       4) Initialize tensors to append decoded words to.
       5) Iteratively decode one word token at a time:
           a) Forward pass through decoder.
           b) Obtain most likely word token and its softmax score.
           c) Record token and score.
           d) Prepare current token to be next decoder input.
       6) Return collections of word tokens and scores.
    """

    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_TOKEN
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_TOKEN
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


def evaluate(encoder, decoder, searcher, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [osp.word_id(sentence.split())]
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
    decoded_words = [osp.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher):
    input_sentence = ''
    while (1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


# Load model if a loadFilename is provided
if loadFilename and os.path.exists(loadFilename):
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    # voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(vocab_size + 1, hidden_size)
if loadFilename and os.path.exists(loadFilename):
    embedding.load_state_dict(embedding_sd)
else:
    pretrained_embeddings = vocab.Vocab(osp.words, vocab_size, 3, vectors="glove.6B.300d",
                                        vectors_cache='../.vector_cache').vectors
    print(vocab_size, pretrained_embeddings.shape)
    embedding.weight.data.copy_(pretrained_embeddings)
    print("Initialized with pre-trained embeddings")

# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, vocab_size, decoder_n_layers, dropout)
if loadFilename and os.path.exists(loadFilename):
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename and os.path.exists(loadFilename):
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)
else:
    print(os.path.exists(loadFilename), loadFilename)
    print("No existing model to begin with")

# Run training iterations
trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_n_layers, decoder_n_layers, clip,
           loadFilename)

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
# evaluateInput(encoder, decoder, searcher)
