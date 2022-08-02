import numpy
from gym.envs import register


def get_terminals(map_size):
    return [(numpy.index_exp[:map_size // 2, map_size // 2:], True)]


def get_rewards(map_size):
    return (
        (numpy.index_exp[:map_size // 2, 3 * map_size // 4:], 1),
        (numpy.index_exp[:map_size // 2, map_size // 2:3 * map_size // 4], -1))


register(
    id="Navigation-v0",
    entry_point="navigation.env:NavigationEnv",
    kwargs=dict(spawn_radius=2, max_ep_len=100, map_size=10, rewards=get_rewards(10), terminals=get_terminals(10))
)

register(
    id="Navigation-v1",
    entry_point="navigation.env:NavigationEnv",
    kwargs=dict(spawn_radius=2, max_ep_len=100, rewards=get_rewards(20), terminals=get_terminals(20))
)

register(
    id="NavigationWide-v0",
    entry_point="navigation.env:NavigationEnv",
    kwargs=dict(spawn_radius=20, max_ep_len=100, map_size=10, rewards=get_rewards(10), terminals=get_terminals(10))
)

register(
    id="NavigationWide-v1",
    entry_point="navigation.env:NavigationEnv",
    kwargs=dict(spawn_radius=20, max_ep_len=100, rewards=get_rewards(20), terminals=get_terminals(20))
)

register(
    id="NavigationNoTerminals-v0",
    entry_point="navigation.env:NavigationEnv",
    kwargs=dict(terminals=None, spawn_radius=20, max_ep_len=100, map_size=10, rewards=get_rewards(10))
)

register(
    id="NavigationNoTerminals-v1",
    entry_point="navigation.env:NavigationEnv",
    kwargs=dict(terminals=None, spawn_radius=20, max_ep_len=100, rewards=get_rewards(20))
)

# register(id="Navigation-v2", entry_point="navigation.env:NavigationEnv", map_size=10,
#          kwargs=dict(terminals=[(numpy.index_exp[8:, 8:], 1)], spawn_radius=20, max_ep_len=100,
#                      rewards=[(numpy.index_exp[8:, 8:], 10), ((numpy.index_exp[3:6, 3:6], -1))]))
import gym

gym.make("Navigation-v0")
