import gym
import numpy as np
import navigation
import pytest

env_names = ["Navigation-v2"]


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


@pytest.mark.parametrize("env_name", env_names)
def test_terminations(env_name):
    # ensures that we get correct termination and reward signals from terminal transitions
    # Uses some gross hacking - will break if the environment is changed too much

    env = gym.make(env_name)

    _ = env.reset()

    # set player position to the right of a terminal position

    # goal state
    env.unwrapped.player = np.array([7,7])  # one below the goal state
    env.unwrapped.player_map = np.zeros((env.unwrapped.h, env.unwrapped.w))
    env.unwrapped.player_map[env.unwrapped.player[0], env.unwrapped.player[1]] = 1

    _, r, d, _ = env.step(np.array([1, 1]).astype("float32"))  # step up into negative reward

    assert r == 10 - 0.01, "Incorrect reward from goal region"
    assert d == 1, "incorrect done signal from goal region"

    env.reset()

    # non terminal neg reward state
    env.unwrapped.player = np.array(
        [2,2])  # one to the left of the goal state
    env.unwrapped.player_map = np.zeros((env.unwrapped.h, env.unwrapped.w))
    env.unwrapped.player_map[env.unwrapped.player[0], env.unwrapped.player[1]] = 1

    _, r, d, _ = env.step(np.array([1, 1]).astype("float32"))  # step right into reward

    assert r == -1 - 0.01, "Incorrect reward from death region"
    assert d == 0, "incorrect done signal from death region"

    env.close()


