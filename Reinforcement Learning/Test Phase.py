import gym
import warnings
warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore", category=DeprecationWarning)
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("CartPole-v1")
env._max_episode_steps = 500
states = env.observation_space.shape[0]  # states [cart position, cart velocity, pole angle, pole angular velocity]
actions = env.action_space.n  # actions [left, right]


def build_model(states, actions):
    model = Sequential()
    # keras.layers.flatten function flattens the multidimensional input tensors into a single dimension
    model.add(Flatten(input_shape=[1, states]))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()  # strategy that an agent uses in pursuit of goals
    memory = SequentialMemory(limit=50000, window_length=1)  # store various states, actions, and rewards
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10,
                   target_model_update=1e-2)
    return dqn


model = build_model(states, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=["mae"])

dqn.load_weights("weights\dqn_weights.h5f")
_ = dqn.test(env, nb_episodes=5, visualize=True)
