from gym.envs import register

register(
    id="Navigation-v0",
    entry_point="navigation.env:NavigationEnv", 
    kwargs=dict(use_terminals=True, spawn_radius=2, max_ep_len=100)
)

register(
    id="NavigationNoTerminals-v0",
    entry_point="navigation.env:NavigationEnv", 
    kwargs=dict(use_terminals=False, spawn_radius=20, max_ep_len=100)
)
