import gym
import navigation
import pytest

def test_make():
    gym.make("Navigation-v0")

def test_render():
    env = gym.make("Navigation-v0")

    _ = env.reset()

    env.render()

def test_episode_no_render():
    env = gym.make("Navigation-v0")

    _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)

def test_episode_render():
    env = gym.make("Navigation-v0")

    _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()