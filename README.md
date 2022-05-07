# Navigation Environment for Gym
A simple 2D navigation environment for testing planning or reinforcement learning algorithms that operate on pixels.

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