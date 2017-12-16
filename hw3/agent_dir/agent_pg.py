from agent_dir.agent import Agent
import os, sys
import scipy
import pickle
import numpy as np
import tensorflow as tf
'''
def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)
'''
# Action values to send to gym environment to move paddle up/down
UP_ACTION = 2
DOWN_ACTION = 3
# Mapping from action values to outputs from the policy network
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards)
    for t in range(len(rewards)):
        discounted_reward_sum = 0
        discount = 1
        for k in range(t, len(rewards)):
            discounted_reward_sum += rewards[k] * discount
            discount *= discount_factor
            if rewards[k] != 0:
                # Don't count rewards from subsequent rounds
                break
        discounted_rewards[t] = discounted_reward_sum
    return discounted_rewards

class Network:
    def __init__(self, hidden_layer_size, learning_rate, checkpoints_dir):
        self.learning_rate = learning_rate

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        #config = tf.ConfigProto( device_count={'CPU' : 1, 'GPU' : 0}, allow_soft_placement=True, log_device_placement=False)
        #self.sess = tf.Session(config=config)
        self.observations = tf.placeholder(tf.float32,
                                           [None, 6400])
        # +1 for up, -1 for down
        self.sampled_actions = tf.placeholder(tf.float32, [None, 1])
        self.advantage = tf.placeholder(
            tf.float32, [None, 1], name='advantage')

        h = tf.layers.dense(
            self.observations,
            units=hidden_layer_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.up_probability = tf.layers.dense(
            h,
            units=1,
            activation=tf.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        # Train based on the log probability of the sampled action.
        # 
        # The idea is to encourage actions taken in rounds where the agent won,
        # and discourage actions in rounds where the agent lost.
        # More specifically, we want to increase the log probability of winning
        # actions, and decrease the log probability of losing actions.
        #
        # Which direction to push the log probability in is controlled by
        # 'advantage', which is the reward for each action in each round.
        # Positive reward pushes the log probability of chosen action up;
        # negative reward pushes the log probability of the chosen action down.
        self.loss = tf.losses.log_loss(
            labels=self.sampled_actions,
            predictions=self.up_probability,
            weights=self.advantage)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(checkpoints_dir,
                                            'policy_network.ckpt')

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    def forward_pass(self, observations):
        up_probability = self.sess.run(
            self.up_probability,
            feed_dict={self.observations: observations.reshape([1, -1])})
        return up_probability

    def train(self, state_action_reward_tuples):
        print("Training with %d (state, action, reward) tuples" %
              len(state_action_reward_tuples))

        states, actions, rewards = zip(*state_action_reward_tuples)
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)

        feed_dict = {
            self.observations: states,
            self.sampled_actions: actions,
            self.advantage: rewards
        }
        self.sess.run(self.train_op, feed_dict)

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.network = Network( 200, 0.0005, checkpoints_dir='checkpoints')
            self.network.load_checkpoint()
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
        np.random.seed(0)
        print('new game')
        observation = prepro(np.zeros((210, 160, 3), dtype=np.int))
        self.last_observation = observation


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        hidden_layer_size = 200
        learning_rate = 0.0005
        batch_size_episodes = 1
        checkpoint_every_n_episodes = 200
        load_checkpoint = False
        discount_factor = 0.99
        network = Network( hidden_layer_size, learning_rate, checkpoints_dir='checkpoints')
        #if load_checkpoint:
        network.load_checkpoint()

        batch_state_action_reward_tuples = []
        smoothed_reward = None
        episode_n = 1

        while True:
            print("Starting episode %d" % episode_n)

            episode_done = False
            episode_reward_sum = 0

            round_n = 1

            last_observation = self.env.reset()
            last_observation = prepro(last_observation)
            action = self.env.action_space.sample()
            observation, _, _, _ = self.env.step(action)
            observation = prepro(observation)
            n_steps = 1

            while not episode_done:
                observation_delta = observation - last_observation
                last_observation = observation
                up_probability = network.forward_pass(observation_delta)[0]
                if np.random.uniform() < up_probability:
                    action = UP_ACTION
                else:
                    action = DOWN_ACTION

                observation, reward, episode_done, info = self.env.step(action)
                observation = prepro(observation)
                episode_reward_sum += reward
                n_steps += 1

                tup = (observation_delta, action_dict[action], reward)
                batch_state_action_reward_tuples.append(tup)

                if reward == -1:
                    print("Round %d: %d time steps; lost..." % (round_n, n_steps))
                elif reward == +1:
                    print("Round %d: %d time steps; won!" % (round_n, n_steps))
                if reward != 0:
                    round_n += 1
                    n_steps = 0

            print("Episode %d finished after %d rounds" % (episode_n, round_n))

            # exponentially smoothed version of reward
            if smoothed_reward is None:
                smoothed_reward = episode_reward_sum
            else:
                smoothed_reward = smoothed_reward * 0.99 + episode_reward_sum * 0.01
            print("Reward total was %.3f; discounted moving average of reward is %.3f" \
                % (episode_reward_sum, smoothed_reward))

            if episode_n % batch_size_episodes == 0:
                states, actions, rewards = zip(*batch_state_action_reward_tuples)
                rewards = discount_rewards(rewards, discount_factor)
                rewards -= np.mean(rewards)
                rewards /= np.std(rewards)
                batch_state_action_reward_tuples = list(zip(states, actions, rewards))
                network.train(batch_state_action_reward_tuples)
                batch_state_action_reward_tuples = []

            if episode_n % checkpoint_every_n_episodes == 0:
                network.save_checkpoint()

            episode_n += 1
            sys.stdout.flush()


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation = prepro(observation)
        observation_delta = observation - self.last_observation
        self.last_observation = observation
        up_probability = self.network.forward_pass(observation_delta)[0]
        if up_probability > np.random.uniform():
            return UP_ACTION
        else:
            return DOWN_ACTION
        #return self.env.get_random_action()

