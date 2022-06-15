"""
A catcher environment reproduced by Cheongwoong Kang.
reference: https://arxiv.org/pdf/1809.04506.pdf
"""
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled


class CatcherEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    Balls that periodically appear at the top of the frames (at random horizontal positions)
    and fall towards the bottom where a paddle (agent) has the possibility to catch them.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction of the agent move.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Left                   |
    | 1   | Right                  |

    ### Observation Space

    The observation is a `ndarray` with shape `(36,36)` with the values corresponding to the 2D map.

    ### Rewards

    The reward is given when the episode ends (+1 for each ball caught, -1 if the ball is not caught).

    ### Starting State

    The ball starts at the far left or the far right.
    The paddle (agent) starts at a random horizontal position.

    ### Episode Termination

    An episode is terminated once the ball reaches the bottom.

    ### Arguments

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 1,
    }

    def __init__(self, rng=np.random.RandomState(123456), render_mode: Optional[str] = None):
        self._random_state = rng
        self._height = 10
        self._width = 10
        self._width_paddle = 1
        self._nx_block = 2

        if self._nx_block == 1:
            self._x_block = self._width // 2
        else:
            rand = self._random_state.randint(self._nx_block)
            self._x_block = rand*((self._width - 1) // (self._nx_block - 1))

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 1, shape=((self._height+2)*3,(self._height+2)*3), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.y = self._height - 1
        self.x = self._random_state.randint(self._width - self._width_paddle + 1)
        if self._nx_block == 1:
            self._x_block = self._width // 2
        else:
            rand = self._random_state.randint(self._nx_block)
            self._x_block = rand*((self._width - 1) // (self._nx_block - 1))

        self.state = self.observe()
        
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        self.act(action)        
        self.state = self.observe()

        if self.y == 0 and self.x > self._x_block - 1 - self._width_paddle and self.x <= self._x_block + 1:
            reward = 1
        elif self.y == 0:
            reward = -1
        else:
            reward = 0
        
        done = True if self.inTerminalState() else False

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def inTerminalState(self):
        if self.y == 0:
            return True
        else:
            return False

    def act(self, action):
        if action == 0:
            self.x = max(self.x - 1, 0)
        elif action == 1:
            self.x = min(self.x + 1, self._width - self._width_paddle)

        self.y = self.y - 1

    def observe(self):
        y_t = (1 + self.y)*3
        x_block_t = (1 + self._x_block)*3
        x_t = (1 + self.x)*3

        obs = np.zeros(( (self._height + 2)*3, (self._width + 2)*3 ))
        ball = np.array([[0,0,0.6,0.8,0.6,0,0],
                            [0.,0.6,0.9,1,0.9,0.6,0],
                            [0.,0.85,1,1,1,0.85,0.],
                            [0,0.6,0.9,1,0.9,0.6,0],
                            [0,0,0.6,0.85,0.6,0,0]])
        paddle = np.array([[0.5,0.95,1,1,1,0.95,0.5],
                            [0.9,1,1,1,1,1,0.9],
                            [0.,0.,0,0,0,0.,0.]])

        obs[y_t-2:y_t+3, x_block_t-3:x_block_t+4] = ball
        obs[3:6, x_t-3:x_t+4] = paddle

        return obs

    def render(self, mode="human"):
        if self.render_mode is not None:
            return None
        else:
            return self._render(mode)

    def _render(self, mode="human"):
        assert mode in self.metadata["render_modes"]
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state is None:
            return None

        self.surf = pygame.Surface(((self._height + 2)*3, (self._width + 2)*3))
        self.surf.fill((255, 255, 255))
        
        for y, x in np.argwhere(self.state > 0):
            gfxdraw.pixel(self.surf, x, y, (0, 0, 0))

        self.surf = pygame.transform.scale(self.surf, (self.screen_width, self.screen_height))
        # self.surf = pygame.transform.rotate(self.surf, 90)
        
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False