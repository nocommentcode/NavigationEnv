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

register(id="Navigation-v2", entry_point="navigation.env:NavigationEnv",
         kwargs=dict(map_size=10, terminals=[(numpy.index_exp[8:, 8:], True),(numpy.index_exp[3:8, 3:7], True)], spawn_radius=1, spawn_center=[1,1], max_ep_len=700,
                     rewards=[(numpy.index_exp[8:, 8:], 1), ((numpy.index_exp[3:8, 3:7], -1))]))

register(id="Navigation-v3", entry_point="navigation.env:NavigationEnv",
         kwargs=dict(map_size=10, terminals=[(numpy.index_exp[8:, 8:], True)], spawn_radius=1, spawn_center=[1,1], max_ep_len=700,
                     rewards=[(numpy.index_exp[8:, 8:], 1), ((numpy.index_exp[3:8, 3:7], -0.05))]))

if __name__ == "__main__":
    import gym
    env = gym.make("Navigation-v3")
    state = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
    g = 9