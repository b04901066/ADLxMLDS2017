from agent_dir.agent import Agent
import os, sys, random
import numpy as np
from collections import deque
import tensorflow as tf

import keras
import keras.losses
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras import backend as K

def _huber_loss(target, prediction):
    # sqrt(1+error^2)-1
    error = prediction - target
    return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
keras.losses._huber_loss = _huber_loss

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.000004
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=8, strides=(4, 4), padding='valid', input_shape=self.state_size, activation='relu'))
        model.add(Conv2D(64, kernel_size=4, strides=(2, 2), padding='valid', activation='relu'))
        model.add(Conv2D(64, kernel_size=3, strides=(1, 1), padding='valid', activation='relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(4, activation='linear'))
        model.compile(loss=self._huber_loss, optimizer=RMSprop(lr=self.learning_rate, rho=0.99, epsilon=1e-08, decay=0.0))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay


    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            self.agent = DQNAgent( (84, 84, 4), 4)
            self.agent.load("./dqn.h5")
            print('loading complete')

        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        agent = DQNAgent( (84, 84, 4), 4)
        # agent.load("./save/cartpole-ddqn.h5")
        done = False
        batch_size = 32
        episode = 0
        total_step = 0
        while True:
            state = self.env.reset()
            state = np.reshape(state, (1, 84, 84, 4))
            total_reward = 0
            episode += 1
            for time in range(10000):
                action = agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                next_state = np.reshape(next_state, (1, 84, 84, 4))
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_step += 1
                if total_step > 10000 and total_step % 4 == 0:
                    agent.replay(batch_size)
                    if total_step % 1000 == 0:
                        agent.update_target_model()
                if done:
                    print("episode, {}, timestep, {}, total_step, {}, total_reward, {}, e, {:.2}"
                           .format(episode, time, total_step, total_reward, agent.epsilon))
                    sys.stdout.flush()
                    break
            if episode % 100 == 0:
                agent.save("./dqn.h5")


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        return np.argmax( self.agent.model.predict( np.reshape(observation, (1, 84, 84, 4)) )[0] )
