## Reinforcement Learning Chatbot

### Types of chatbots
- General ( Chat on an array of topics like sports, politics, religion, hobbies, movies)
- Goal Oriented ( Used to book holidays, parts of alexa or google home mini used to play songs, set reminders, etc). Currently they are limited in that they do not generalize well across domains.

Chatbot lends naturally itself to the Reinforcement Learning Paradigm. The objective to maximize is the length of the dialog that chatbots hold with humans or maybe fellow chatbot.

### Reinforement Learning Algorithms to try
- Policy Gradient
- and more

### Room for innovation
- The chatbots will get better. We've to try the alternatives.

### Technology Stack
- numpy=1.6.1
- pytorch=1.0.1 with GPU
- python=3.6.7

### Datasets
- OpenSubtitle 80 million dialog (Naturally all of them are not our everyday chats. )

### Reference Papers
- Deep Reinforcement Learning for Dialogue Generation
- A Deep Reinforcement Learning Chatbot
- Building Advanced Dialogue Managers for Goal-Oriented Dialogue Systems

### RL model for chatbot
Assume we have two chatbots talking to each other. The chats proceed as follows : p1, q1, p2, q2, ..

- Policy : parameters of the encoder / decoder LSTM
- State : p1, q1
- Action : p2 which is an arbitrary length dialog
- Reward : how was the reponse ( drives chat further, repeatitive, coherent, mututal information )


### The Next Two Weeks
- Figure out Policy Gradient (Apply it on one of Gym's environment)
- Preprocessing data
- Setup GCP 
