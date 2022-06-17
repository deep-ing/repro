"""
A simple labyrinth environment reproduced by Cheongwoong Kang.
reference: https://arxiv.org/pdf/1809.04506.pdf
"""
import copy
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled


class SimpleMazeEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    The agent moves in the four cardinal directions (by 6 pixels) thanks to the four possible actions, except when the agent reaches a wall (block of 6 X 6 black pixels).

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1, 2, 3}` indicating the direction of the agent move.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Up                     |
    | 1   | Down                   |
    | 2   | Left                   |
    | 3   | Right                  |

    ### Observation Space

    The observation is a `ndarray` with shape `(48,48)` with the values corresponding to the 2D map.

    ### Rewards

    This simple labyrinth environment has no reward.

    ### Starting State

    The agent starts at (2,2).

    ### Episode Termination

    This environment has no terminal state.

    ### Arguments

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 1,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self._size_maze = 8
        self.create_map()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, shape=(self._size_maze*6,self._size_maze*6), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

    def create_map(self):
        self._map = np.zeros((self._size_maze, self._size_maze))
        self._map[0,:] = 1
        self._map[-1,:] = 1
        self._map[:,0] = 1
        self._map[:,-1] = 1

        self._map[:,self._size_maze//2] = 1
        self._map[self._size_maze//2,self._size_maze//2] = 0

        self._pos_agent = [self._size_maze//2,self._size_maze//2]
        self._pos_goal = [self._size_maze-2, self._size_maze-2]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.create_map()

        self._pos_agent = [self._size_maze//2,self._size_maze//2]

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

        done = False
        reward = 0
        
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def act(self, action):
        self._cur_action = action
        if action == 0:
            if self._map[self._pos_agent[0]-1,self._pos_agent[1]] == 0:
                self._pos_agent[0] = self._pos_agent[0] - 1
        elif action == 1:
            if self._map[self._pos_agent[0]+1,self._pos_agent[1]] == 0:
                self._pos_agent[0] = self._pos_agent[0] + 1
        elif action == 2:
            if self._map[self._pos_agent[0],self._pos_agent[1]-1] == 0:
                self._pos_agent[1] = self._pos_agent[1] - 1
        elif action == 3:
            if self._map[self._pos_agent[0],self._pos_agent[1]+1] == 0:
                self._pos_agent[1] = self._pos_agent[1] + 1

    def observe(self):
        obs = copy.deepcopy(self._map)

        obs = obs / 1.
        obs = np.repeat(np.repeat(obs, 6, axis=0), 6, axis=1)

        # agent repr
        agent_obs = np.zeros((6,6))
        agent_obs[0,2] = 0.7
        agent_obs[1,0:5] = 0.8
        agent_obs[2,1:4] = 0.8
        agent_obs[3,1:4] = 0.8
        agent_obs[4,1] = 0.8
        agent_obs[4,3] = 0.8
        agent_obs[5,0:2] = 0.8
        agent_obs[5,3:5] = 0.8
        
        # reward repr
        reward_obs = np.zeros((6,6))
        #reward_obs[:,1] = 0.8
        #reward_obs[0,1:4] = 0.7
        #reward_obs[1,3] = 0.8
        #reward_obs[2,1:4] = 0.7
        #reward_obs[4,2] = 0.8
        #reward_obs[5,2:4] = 0.8

        i = self._pos_goal
        obs[i[0]*6:(i[0]+1)*6:,i[1]*6:(i[1]+1)*6] = reward_obs

        i = self._pos_agent
        obs[i[0]*6:(i[0]+1)*6:,i[1]*6:(i[1]+1)*6] = agent_obs
            
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

        self.surf = pygame.Surface((self._size_maze*6, self._size_maze*6))
        self.surf.fill((255, 255, 255))
        
        for x, y in np.transpose(self.state.nonzero()):
            gfxdraw.pixel(self.surf, x, y, (0, 0, 0))

        self.surf = pygame.transform.scale(self.surf, (self.screen_width, self.screen_height))
        self.surf = pygame.transform.rotate(self.surf, 90)
        
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