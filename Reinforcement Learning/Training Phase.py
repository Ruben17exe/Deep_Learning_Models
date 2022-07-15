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


# Create environment with OpenAI Gym
env = gym.make("CartPole-v1")
env._max_episode_steps = 500
states = env.observation_space.shape[0]  # states [cart position, cart velocity, pole angle, pole angular velocity]
actions = env.action_space.n  # actions [left, right]
# ----------------------------------------------------------------------


# Create a Deep Learning Model with Keras
def build_model(states, actions):
    model = Sequential()
    # keras.layers.flatten function flattens the multidimensional input tensors into a single dimension
    model.add(Flatten(input_shape=[1, states]))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


model = build_model(states, actions)
# ----------------------------------------------------------------------


# Build Agent with Keras-RL
def build_agent(model, actions):
    policy = BoltzmannQPolicy()  # strategy that an agent uses in pursuit of goals
    memory = SequentialMemory(limit=50000, window_length=1)  # store various states, actions, and rewards
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                   nb_actions=actions, nb_steps_warmup=10,
                   target_model_update=1e-2)
    return dqn


dqn = build_agent(model, actions)

dqn.compile(Adam(lr=1e-3), metrics=["mae"])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
# ----------------------------------------------------------------------


dqn.save_weights("weights\dqn_weights.h5f", overwrite=True)
# ----------------------------------------------------------------------
