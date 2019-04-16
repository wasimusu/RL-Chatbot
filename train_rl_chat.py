from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from torchtext import vocab

from dataloader import OpenSubtitle
from network import *
from rl import *
import config

config = config.Config()

# Configure training/optimization
clip = 50.0
decoder_learning_ratio = 5.0

# Configure models
attn_model = 'dot'
encoder_num_layers = 2
decoder_num_layers = 2
dropout = 0.1

# Set checkpoint to load from; set to None if starting from scratch
dir = os.path.join(config.save_dir, config.corpus_name,
                   '{}-{}_{}'.format(encoder_num_layers, decoder_num_layers, config.hidden_size))
if not os.path.exists(dir): os.makedirs(dir)
checkpoints = os.listdir(dir)
checkpoints = [int(filename.split('_')[0]) for filename in checkpoints]
if checkpoints == []:
    checkpoints = 1
else:
    checkpoints = max(checkpoints)
loadFilename = os.path.join(dir, '{}_checkpoint.tar'.format(checkpoints))
print(loadFilename, os.path.exists(loadFilename))

osp = OpenSubtitle(config.filename, batch_size=config.batch_size, vocab_size=config.vocab_size, shuffle=True)
data = osp.next()
inputs, input_lengths, targets, mask, max_target_length = data

SOS_TOKEN = osp.word2index['<sos>']
EOS_TOKEN = osp.word2index['<eos>']
PAD_TOKEN = osp.word2index['<pad>']
vocab_size = len(osp.vocab)  # Because some additional words were added to the vocab
print("Vocab size : ", vocab_size)


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, encoder_optimizer,
          decoder_optimizer, clip):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set config.device options
    input_variable = input_variable.to(config.device)
    lengths = lengths.to(config.device)
    target_variable = target_variable.to(config.device)
    mask = mask.to(config.device)

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
    decoder_input = decoder_input.to(config.device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.num_layers]
    # print("Here we are using only last state of decoder for encoder's hidden state : ", decoder_hidden.size())

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < config.teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, *_ = decoder(
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
            decoder_input = decoder_input.to(config.device)
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


def trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_num_layers, decoder_num_layers,
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
            if iteration % config.print_every == 0:
                print_loss_avg = print_loss / config.print_every
                print("Epoch: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(epoch,
                                                                                          iteration / len(
                                                                                              osp) * 100,
                                                                                          print_loss_avg))
                print_loss = 0

            # Save checkpoint
            if (iteration % config.save_every == 0):
                directory = os.path.join(config.save_dir, config.corpus_name,
                                         '{}-{}_{}'.format(encoder_num_layers, decoder_num_layers, config.hidden_size))
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


def evaluate(encoder, decoder, searcher, sentence, max_length=config.max_sentence_len):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [osp.word_id(sentence.split())]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate config.device
    input_batch = input_batch.to(config.device)
    lengths = lengths.to(config.device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [osp.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher):
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
    # checkpoint = torch.load(loadFilename, map_location=torch.config.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']

# Initialize word embeddings
embedding = nn.Embedding(vocab_size, config.hidden_size)
if loadFilename and os.path.exists(loadFilename):
    embedding.load_state_dict(embedding_sd)
else:
    pretrained_embeddings = vocab.Vocab(osp.words, vocab_size, min_freq=3,
                                        vectors="glove.6B.{}d".format(config.hidden_size), specials=[],
                                        vectors_cache=config.vectors_cache).vectors
    assert vocab_size == pretrained_embeddings.shape[0]
    print(embedding.weight.data.size(), pretrained_embeddings.shape)
    print("Initialized with pre-trained embeddings")

print("Declared vocab size of decoder : ", vocab_size)
# All the below listed encoders and decoders are compatible with each other
# Initialize encoder & decoder models
encoder = EncoderRNN(config.hidden_size, embedding, encoder_num_layers, dropout, config.batch_size)
# decoder = LuongAttnDecoderRNN(attn_model, embedding, config.hidden_size, vocab_size, decoder_num_layers, dropout)
decoder = PlainDecoder(attn_model, embedding, config.hidden_size, vocab_size, config.batch_size, decoder_num_layers)
# decoder = RLDecoder(config.hidden_size, vocab_size)  # TODO : It's hacky. Not a good way to do it.
# TODO : Mysteriously it can't load citing different embedding dim
if loadFilename and os.path.exists(loadFilename):
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

# Use appropriate config.device
encoder = encoder.to(config.device)
decoder = decoder.to(config.device)

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate * decoder_learning_ratio)
if loadFilename and os.path.exists(loadFilename):
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)
else:
    print(os.path.exists(loadFilename), loadFilename)
    print("No existing model to begin with")

# # Train Seq2Seq
# trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_num_layers, decoder_num_layers, clip,
#            loadFilename)

# Train RL
batch_size = 1
# Forward
forward_encoder = EncoderRNN(config.hidden_size, embedding, encoder_num_layers, dropout, batch_size)
forward_decoder = PlainDecoder(attn_model, embedding, config.hidden_size, vocab_size, batch_size, decoder_num_layers,
                               dropout)

# Backward
backward_encoder = EncoderRNN(config.hidden_size, embedding, encoder_num_layers, dropout, batch_size)
backward_decoder = PlainDecoder(attn_model, embedding, config.hidden_size, vocab_size, batch_size, decoder_num_layers,
                                dropout)

if config.use_cuda:
    forward_encoder = forward_encoder.to(config.device)
    forward_decoder = forward_decoder.to(config.device)

    backward_encoder = backward_encoder.to(config.device)
    backward_decoder = backward_decoder.to(config.device)

print("Training RL network")
dull_responses = []
trainRLIters(forward_encoder, forward_decoder, backward_encoder, backward_decoder, dull_responses,
             n_iters=10000, print_every=100, learning_rate=0.0001,
             teacher_forcing_ratio=0.5, bidirectional=False)

# Finally you can chat with the agent
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher)
