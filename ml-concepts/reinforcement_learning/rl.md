# Reinforcement Learning

<aside>
🧩 The main idea behind reinforcement learning is that it’s used for robotics. Pieter Abbeel gave a demo of robots doing human-like tasks, but they were all human controlled. Robots are now physically capable of doing all these tasks, the motivation behind RL is to embed them with the necessary intelligence to carry out these tasks on their own.

</aside>

- In ML, supervised learning is used a lot. You give an input to a model, but you know the output that your model should produce, so using backpropagation, the model can adjust gradients to increase accuracy of the model
- For ex. if we take the example of a game of pong, if we train the model on the gameplay of the best pong player, then the model would simply be imitating the actions.
    - In other words, the model would never be better than the human player
    - And another downside is that creating data is very hard
- If the goal is to get the model to be better than the human, then supervised learning cannot be used
- RL is where we’re concerned with how software agents take actions in an environment in order to maximize reward
    - By maximizing reward, we allow the model to learn better and pick up all the quirks of how to beat the task / game
- **deep Q learning**
    - this approach extends RL by using a deep neural network to predict the actions