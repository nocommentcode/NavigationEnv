# Navigation Environment for Gym
A simple 2D navigation environment for testing planning or reinforcement learning algorithms that operate on pixels.

The agent (white dot) must navigate to the reward region (green) without hitting and of the bad region (red). Episodes
terminate at the arrival to the red or green zones, or after a set amount of steps. Initial state will be somewhere in the
bottom right region of the board - the exact position is random. All transitions are deterministic (for now).

![An image of the environment](media/navigation.png)

## Installation
```
git clone https://github.com/will-maclean/NavigationEnv.git
cd NavigationEnv
pip install -e .
```

## Usage
For a simple example of the environment:

```python
import gym
import navigation

gym.make("Navigation-v0")

state = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    env.render()
```