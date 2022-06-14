"""
A labyrinth environment reproduced by Cheongwoong Kang.
reference: https://arxiv.org/pdf/1809.04506.pdf
"""
import copy
from .a_star_path_finding import AStar
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled


class MazeEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
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

    The observation is a `ndarray` with shape `(8,8)` with the values corresponding to the 2D map.

    ### Rewards

    The reward is given when the agent reaches one of the three keys that are randomly placed in the map.

    ### Starting State

    The agent starts at (1,1).

    ### Episode Termination

    An episode is terminated once all rewards have been gathered by the agent.
    In the test phase, the episode is also terminated when the number of 50 steps has been reached.

    ### Arguments

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 1,
    }

    def __init__(self, rng=np.random.RandomState(123456), render_mode: Optional[str] = None):
        self._random_state = rng
        self._size_maze = 8
        self._n_walls = int((self._size_maze-2)**2/3.)
        self._n_rewards = 3
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
        valid_map = False
        while valid_map == False:
            # Agent
            self._pos_agent = [1,1]

            # Walls
            self._pos_walls = []
            for i in range(self._size_maze):
                self._pos_walls.append([i,0])
                self._pos_walls.append([i,self._size_maze-1])
            for j in range(self._size_maze - 2):
                self._pos_walls.append([0,j+1])
                self._pos_walls.append([self._size_maze-1,j+1])
            
            n = 0
            while n < self._n_walls:
                potential_wall = [self._random_state.randint(1, self._size_maze - 2), self._random_state.randint(1, self._size_maze - 2)]
                if potential_wall not in self._pos_walls and potential_wall != self._pos_agent:
                    self._pos_walls.append(potential_wall)
                    n += 1
            
            # Rewards
            #self._pos_rewards=[[self._size_maze-2,self._size_maze-2]]
            self._pos_rewards = []
            n = 0
            while n < self._n_rewards:
                potential_reward = [self._random_state.randint(1, self._size_maze - 1), self._random_state.randint(1, self._size_maze - 1)]
                if potential_reward not in self._pos_rewards and potential_reward not in self._pos_walls and potential_reward != self._pos_agent:
                    self._pos_rewards.append(potential_reward)
                    n += 1
            
            valid_map = self.is_valid_map(self._pos_agent, self._pos_walls, self._pos_rewards)

    def is_valid_map(self, pos_agent, pos_walls, pos_rewards):
        a = AStar()
        walls = [tuple(w) for w in pos_walls]
        start = tuple(pos_agent)
        for r in pos_rewards:
            end = tuple(r)
            a.init_grid(self._size_maze, self._size_maze, walls, start, end)
            maze = a
            optimal_path = maze.solve()
            if optimal_path == None:
                return False
        return True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.create_map()

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

        reward = -0.1
        if self._pos_agent in self._pos_rewards:
            reward = 1
            self._pos_rewards.remove(self._pos_agent)
        done = True if self.inTerminalState() else False

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def inTerminalState(self):
        if self._pos_rewards == []:
            return True
        else:
            return False

    def act(self, action):
        self._cur_action = action
        if action == 0:
            if([self._pos_agent[0]-1,self._pos_agent[1]] not in self._pos_walls):
                self._pos_agent[0] = self._pos_agent[0] - 1
        elif action == 1:
            if([self._pos_agent[0]+1,self._pos_agent[1]] not in self._pos_walls):
                self._pos_agent[0] = self._pos_agent[0] + 1
        elif action == 2:
            if([self._pos_agent[0],self._pos_agent[1]-1] not in self._pos_walls):
                self._pos_agent[1] = self._pos_agent[1] - 1
        elif action == 3:
            if([self._pos_agent[0],self._pos_agent[1]+1] not in self._pos_walls):
                self._pos_agent[1] = self._pos_agent[1] + 1

    def observe(self):
        self._map = np.zeros((self._size_maze, self._size_maze))
        for x, y in self._pos_walls:
            self._map[x, y] = 1
        for x, y in self._pos_rewards:
            self._map[x, y] = 2
        self._map[self._pos_agent[0], self._pos_agent[1]] = 0.5

        indices_reward = np.argwhere(self._map == 2)
        indices_agent = np.argwhere(self._map == 0.5)
        self._map = self._map / 1.
        self._map = np.repeat(np.repeat(self._map, 6, axis=0), 6, axis=1)

        # agent repr
        agent_obs = np.zeros((6,6))
        agent_obs[0,2] = 0.8
        agent_obs[1,0:5] = 0.9
        agent_obs[2,1:4] = 0.9
        agent_obs[3,1:4] = 0.9
        agent_obs[4,1] = 0.9
        agent_obs[4,3] = 0.9
        agent_obs[5,0:2] = 0.9
        agent_obs[5,3:5] = 0.9
        
        # reward repr
        reward_obs = np.zeros((6,6))
        reward_obs[:,1] = 0.7
        reward_obs[0,1:4] = 0.6
        reward_obs[1,3] = 0.7
        reward_obs[2,1:4] = 0.6
        reward_obs[4,2] = 0.7
        reward_obs[5,2:4] = 0.7

        for i in indices_reward:
            self._map[i[0]*6:(i[0]+1)*6:,i[1]*6:(i[1]+1)*6] = reward_obs

        for i in indices_agent:
            self._map[i[0]*6:(i[0]+1)*6:,i[1]*6:(i[1]+1)*6] = agent_obs
        
        self._map = (self._map*2) - 1

        return self._map

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
        
        for x, y in np.argwhere(self.state == 1):
            gfxdraw.pixel(self.surf, x, y, (0, 0, 0))

        for x, y in np.argwhere((self.state > 0.5) & (self.state < 1.0)):
            gfxdraw.pixel(self.surf, x, y, (0, 0, 0))

        for x, y in np.argwhere((self.state > 0.0) & (self.state < 0.5)):
            gfxdraw.pixel(self.surf, x, y, (255, 180, 0))

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