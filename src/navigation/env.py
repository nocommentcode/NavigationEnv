import gym
import numpy as np
import matplotlib.pyplot as plt

class NavigationEnv(gym.Env):
    metadata = {"render.modes": ["human", "array"]}

    def __init__(self, use_terminals=False, spawn_radius=2, max_ep_len=100, map_size=20):
        super().__init__()
        self.w = map_size
        self.h = map_size

        # create reward map
        self.rewards = np.zeros((self.h, self.w))
        self.rewards[:map_size//4, :map_size//2] = 1
        self.rewards[map_size//4:map_size//2, :map_size//2] = -1

        self.timestep_penalty = 0.01

        # create terminal map
        self.terminals = np.zeros((self.h, self.w))
        self.terminals[:map_size//2, :map_size//2] = use_terminals

        self.max_ep_len = max_ep_len
        self.current_ep_len = 0
        self.ep_return = 0

        self.player = None
        self.player_map = np.zeros((self.h, self.w))
        self.spawn_center = (3 * map_size//2, 3 * map_size//2)
        self.spawn_radius = spawn_radius

        # channel 1: player location
        # channel 2: terminal map
        # channel 3: reward map
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3, self.h, self.w))

        # N, E, S, W
        self.action_space = gym.spaces.Discrete(4)

        # Visualisation stuff
        self.visualized = False
        self.closed = False

    def reset(self):
        self.current_ep_len = 0
        self.ep_return = 0

        random_shift_x = 2 * np.random.randint(self.spawn_radius + 1) - self.spawn_radius
        random_shift_y = 2 * np.random.randint(self.spawn_radius + 1) - self.spawn_radius

        self.player = np.array([self.spawn_center[1] + random_shift_y, self.spawn_center[0] + random_shift_x])
        self.player[0] = np.clip(self.player[0], 0, self.h - 1)
        self.player[1] = np.clip(self.player[1], 0, self.w - 1)

        self.player_map = np.zeros((self.h, self.w))
        self.player_map[self.player[0], self.player[1]] = 1

        return self._obs()
    
    def step(self, action):
        assert self.action_space.contains(action)

        self.current_ep_len += 1
    
        # Fill this up with any extra info required
        info = {
            "ep_len": self.current_ep_len,
            'ep_return': self.ep_return,
        }

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
        self.player[0] = np.clip(self.player[0], 0, self.h - 1)
        self.player[1] = np.clip(self.player[1], 0, self.w - 1)

        self.player_map = np.zeros((self.h, self.w))
        self.player_map[self.player[0], self.player[1]] = 1

        r = self.rewards[self.player[0], self.player[1]] - self.timestep_penalty
        d = self.terminals[self.player[0], self.player[1]] or self.current_ep_len >= self.max_ep_len

        self.ep_return += r

        return self._obs(), r, d, info

    def _obs(self):
        return np.stack([self.player_map, self.rewards, self.terminals])
    
    def render(self, mode="human", time=50):
        state = self._obs()

        pixels = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for x in range(state.shape[1]):
            for y in range(state.shape[2]):
                if state[0, x, y] == 1:
                    # player
                    pixels[x, y] = np.array([255, 255, 255])  # white

                elif state[1, x, y] == 1:
                    # rewards
                    pixels[x, y] = np.array([0, 255, 0])  # green

                elif state[1, x, y] == -1:
                    # bad reward
                    pixels[x, y, 0] = 255  # red
        
        if mode == "human":
            if not self.visualized:
                global plt
                mpl = __import__('matplotlib.pyplot', globals(), locals())
                plt = mpl.pyplot
                _, self.ax = plt.subplots(1,1)
                plt.show(block=False)
                self.visualized = True
            if self.closed:
                _, self.ax = plt.subplots(1,1)
                plt.show(block=False)
                self.closed = False
            
            self.ax.imshow(pixels)
            self.ax.set_title(f"Return: {self.ep_return:.2f}")
            plt.pause(time/1000)
            plt.cla()
        
        elif mode == "rgb_array":
            return pixels
        
    def close_display(self):
        plt.close()
        self.closed = True
    
    def close(self):
        super().close()
        self.close_display()

