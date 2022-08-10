import numpy as np

from navigation.env import NavigationEnv


class RandomGenerationNavigationEnv(NavigationEnv):
    """
    A navigation environment that can randomise its goal and reward areas
    """

    def __init__(self, fixed_elements, random_elements, spawn_radius=2,
                 spawn_center=(3 * 20 // 2, 3 * 20 // 2), max_ep_len=100, map_size=20):
        """
        Initialise the environment
        :param fixed_elements: the fixed elements that are always the same (array of (numpy_slice_of_map, reward, terminal))
        :param random_elements: the elements that should be randomised each instance (array of (width, height, reward, terminal))
        :param spawn_radius: the radius to spawn the agent from
        :param spawn_center: the center to spawn the agent from
        :param max_ep_len: the maximum number of timesteps
        :param map_size: the size of the map
        """
        self.fixed_elements = fixed_elements
        self.random_elements = random_elements

        super().__init__([], [], spawn_radius, spawn_center, max_ep_len, map_size)
        self.randomise_instance()
        self.randomisation_locked = False

    def match_to_other_instance(self, env):
        self.terminals = env.terminals.copy()
        self.rewards = env.rewards.copy()

    def lock_randomisation(self):
        self.randomisation_locked = True

    def unlock_randomisation(self):
        self.randomisation_locked = False

    def reset(self):
        if not self.randomisation_locked:
            self.randomise_instance()
        super().reset()

    def randomise_instance(self):
        # clear terminals and rewards
        self.terminals = np.zeros((self.h, self.w))
        self.rewards = np.zeros((self.w, self.h))

        # handle fixed elements
        for s, r, t in self.fixed_elements:
            self.terminals[s] = t
            self.rewards[s] = r

        # handle random elements
        for w, h, r, t in self.random_elements:
            found_bottom_corner = False
            corner = None

            while not found_bottom_corner:
                # randomly select bottom left corner
                corner = np.random.randint([0, 0], [self.w - w, self.h - h], (2))

                # ensure no collision with other elements
                collision_occured = False
                for x in range(w + 1):
                    for y in range(h + 1):
                        if not self.terminals[corner[0] + x, corner[1] + y] == 0 and self.rewards[
                            corner[0] + x, corner[1] + y] == 0:
                            collision_occured = True

                found_bottom_corner = not collision_occured

            self.terminals[corner[0]:corner[0] + w, corner[1]:corner[1] + h] = t
            self.rewards[corner[0]:corner[0] + w, corner[1]:corner[1] + h] = r
