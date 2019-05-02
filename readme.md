## Reinforcement Learning Chatbot

### Types of chatbots
- General ( Chat on an array of topics like sports, politics, religion, hobbies, movies)
- Goal Oriented ( Used to book holidays, parts of alexa or google home mini used to play songs, set reminders, etc). Currently they are limited in that they do not generalize well across domains.

Chatbot lends naturally itself to the Reinforcement Learning Paradigm. The objective to maximize is the length of the dialog that chatbots hold with humans or maybe fellow chatbot.

### Datasets
- Works with OpenSubtitle 80 million dialog (Naturally all of them are not our everyday chats. )
- Cornell Dialog Corpus used for test architecture because it's small and thus faster

### RL model for chatbot
Assume we have two chatbots talking to each other. The chats proceed as follows : p1, q1, p2, q2, ..

- Policy : parameters of the encoder / decoder LSTM
- State : p1, q1
- Action : p2 which is an arbitrary length dialog
- Reward : weighted sum of semantic coherence, ease of answering and information flow


### Training the chatbot using Policy Gradient
- First train the Seq2Seq network to generate response given a dialog.
- Initiate two agents : one network to generate response given dialog and another to generate input given current response.
- One agent keeps talking given starting dialog. The responses are then p1, q1, p2, q2 . . . . . 
- Compute the reward and back-propagate the network

### Technology Stack
- numpy=1.6.1
- pytorch=1.0.1
- python=3.6.7
- torchtext


### Reference Papers
- Deep Reinforcement Learning for Dialogue Generation
- A Deep Reinforcement Learning Chatbot
- Building Advanced Dialogue Managers for Goal-Oriented Dialogue Systems
