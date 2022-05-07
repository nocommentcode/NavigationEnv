from gym.envs import register

def register_envs():
    register(id="Navigation-v0", entry_point="navigation.env:NavigationEnv")
