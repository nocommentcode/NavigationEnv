import gym
import numpy as np
import navigation
import pytest

env_names = ["Navigation-v0", "Navigation-v1", "NavigationNoTerminals-v0", "NavigationNoTerminals-v1"]

@pytest.mark.parametrize("env_name", env_names)
def test_make(env_name):
    gym.make(env_name)

@pytest.mark.parametrize("env_name", env_names)
def test_render(env_name):
    env = gym.make(env_name)

    _ = env.reset()

    env.render()

    env.close()

@pytest.mark.parametrize("env_name", env_names)
def test_episode_no_render(env_name):
    env = gym.make(env_name)

    _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
    
    env.close()

@pytest.mark.parametrize("env_name", env_names)
def test_episode_render(env_name):
    env = gym.make(env_name)

    _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()
    
    env.close()

@pytest.mark.parametrize("env_name", env_names)
def test_bad_actions(env_name):

    env = gym.make(env_name)

    bad_actions = [np.array([-2, -2]), np.array([-3, 0]), np.array(0), [0, 0]]

    for a in bad_actions:
        with pytest.raises(Exception):
            env.step(a)
    
    env.close()
