import gym
import numpy as np
import navigation
import pytest

terminal_env_names = ["Navigation-v0", "Navigation-v1", "NavigationWide-v0", "NavigationWide-v1"]
non_terminal_env_names = ["NavigationNoTerminals-v0", "NavigationNoTerminals-v1"]
env_names = terminal_env_names + non_terminal_env_names

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

    bad_actions = [-1, 1.5, 4]

    for a in bad_actions:
        with pytest.raises(Exception):
            env.step(a)
    
    env.close()



@pytest.mark.parametrize("env_name", terminal_env_names)
def test_terminations(env_name):
    # ensures that we get correct termination and reward signals from terminal transitions
    # Uses some gross hacking - will break if the environment is changed too much

    env = gym.make(env_name)

    _ = env.reset()

    # set player position to the right of a terminal position
    
    # goal state
    env.unwrapped.player = np.array([0, env.unwrapped.w // 2])  # one to the right of the goal state
    env.unwrapped.player_map = np.zeros((env.unwrapped.h, env.unwrapped.w))
    env.unwrapped.player_map[env.unwrapped.player[0], env.unwrapped.player[1]] = 1

    _, r, d, _ = env.step(3)  # step left into reward

    assert r == 1 - 0.01, "Incorrect reward from goal region"
    assert d == 1, "incorrect done signal from goal region"

    env.reset()

    # terminal state
    env.unwrapped.player = np.array([env.unwrapped.h // 2 - 1, env.unwrapped.w // 2])  # one to the right of the death state
    env.unwrapped.player_map = np.zeros((env.unwrapped.h, env.unwrapped.w))
    env.unwrapped.player_map[env.unwrapped.player[0], env.unwrapped.player[1]] = 1

    _, r, d, _ = env.step(3)  # step left into reward

    assert r == -1 - 0.01, "Incorrect reward from death region"
    assert d == 1, "incorrect done signal from death region"
    
    
    env.close()


@pytest.mark.parametrize("env_name", env_names)
def test_non_terminal_transition_reward(env_name):
    # ensures that we get correct termination and reward signals from terminal transitions
    # Uses some gross hacking - will break if the environment is changed too much

    env = gym.make(env_name)

    _ = env.reset()

    # set player position to bottom right 
    
    # goal state
    env.unwrapped.player = np.array([env.unwrapped.h - 1, env.unwrapped.w - 1])
    env.unwrapped.player_map = np.zeros((env.unwrapped.h, env.unwrapped.w))
    env.unwrapped.player_map[env.unwrapped.player[0], env.unwrapped.player[1]] = 1

    _, r, d, _ = env.step(0)  # step up - non terminal transition

    assert r == -0.01, "Incorrect reward from goal region"
    assert d == 0, "incorrect done signal from goal region"
    
    
    env.close()
