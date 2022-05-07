import gym
import numpy as np

class NavigationEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.w = 20
        self.h = 20

        # create reward map
        self.rewards = np.zeros((self.h, self.w))
        self.rewards[:5, :10] = 1
        self.rewards[5:10, :10] = -1

        # create terminal map
        self.terminals = np.zeros((self.h, self.w))
        self.termingals[:10, :10] = 1

        self.max_ep_len = 100

        self.player = None
        self.player_map = np.zeros((self.h, self.w))
        self.spawn_center = (17, 17)
        self.spawn_radius = 2

        # channel 1: player location
        # channel 2: terminal map
        # channel 3: reward map
        self.observation_space = np.spaces.Box(0.0, 1.0, shape=(3, self.h, self.w))

        # N, E, S, W
        self.action_space = np.spaces.Discrete(4)

    def reset(self):
        self.player = np.array([self.spawn_center[1] + np.random.randint(self.spawn_radius + 1), self.spawn_center[0] + np.random.randint(self.spawn_radius + 1)])
        self.player_map = np.zeros((self.h, self.w))
        self.player_map[self.player[0], self.player[1]] = 1

        return self._obs()
    
    def step(self, action):
        assert self.action_space.contains(action)
    
        # Fill this up with any extra info required
        info = {}

        if action == 0:
            # N
            self.player[0] -= 1
        elif action == 1:
            # E
            self.player[1] += 1
        elif action == 2:
            # S
            self.player[0] += 1
        elif action == 3:
            # W
            self.player[1] -= 1

        # Clip to ensure the player stays on the board
        self.player[0] = np.clip(self.player[0], 0, self.h)
        self.player[1] = np.clip(self.player[1], 0, self.w)

        self.player_map = np.zeros((self.h, self.w))
        self.player_map[self.player[0], self.player[1]] = 1

        r = self.rewards[self.player[0], self.player[1]]
        d = self.terminals[self.player[0], self.palyer[1]]

        return self._obs(), r, d, info

    def _obs(self):
        return np.stack([self.player_map, self.rewards, self.terminals])
