"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import matplotlib

import numpy as np

import gymnasium as gym
from gym import spaces
from gym.utils import seeding

from gym.envs.classic_control.mountain_car import MountainCarEnv as MCE



class MountainCarEnv(MCE):
    def __init__(self, discount=0.99, goal_velocity=0):
        super().__init__(goal_velocity=goal_velocity)
        self.discount = discount
        self.max_path_length = 500
        self.vectorized = True

    def step(self, action):
        return super().step(action)[:-1]

    def render(self, mode='human', iteration=None):
        self.render_mode = mode
        return super().render()

    #def __init__(self, discount=0.99, goal_velocity=0):
    #    self.min_position = -1.2
    #    self.max_position = 0.6
    #    self.max_speed = 0.07
    #    self.goal_position = 0.5
    #    self.goal_velocity = goal_velocity
    #    self.discount = discount
    #    self.max_path_length = 500
    #    self.dt = 0.005

    #    self.force = 0.001
    #    self.gravity = 0.0025

    #    self.low = np.array([self.min_position, -self.max_speed])
    #    self.high = np.array([self.max_position, self.max_speed])

    #    self.viewer = None

    #    self.action_space = spaces.Discrete(3)
    #    self.observation_space = spaces.Box(self.low, self.high, dtype=np.float64)
    #    self.vectorized = True

    #    self.seed()

    #def seed(self, seed=None):
    #    self.np_random, seed = seeding.np_random(seed)
    #    return [seed]

    #def step(self, action):
    #    if isinstance(action, np.ndarray):
    #        assert action.shape == (1,)
    #        action = action[0]
    #    assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

    #    position, velocity = self.state
    #    velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
    #    velocity = np.clip(velocity, -self.max_speed, self.max_speed)
    #    position += velocity
    #    position = np.clip(position, self.min_position, self.max_position)
    #    if (position == self.min_position and velocity < 0): velocity = 0

    #    done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
    #    reward = -1.0

    #    self.state = (position, velocity)
    #    return np.array(self.state).copy(), reward, done, {}

    def reset(self):
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        # self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

        # if self.render_mode == "human":
        #     self.render()
        # return np.array(self.state, dtype=np.float32), {}

        # self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        self.state = np.array([-0.5, 0.])
        return np.array(self.state).copy()

    def set_state(self, state):
        self.state = state

    #def _height(self, xs):
    #    return np.sin(3 * xs) * .45 + .55

    def vec_step(self, actions):
        assert self._states is not None

        position, velocity = (self._states.T).copy()
        velocity += (actions - 1) * self.force + np.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        position[(position == self.min_position) * (velocity < 0)] = 0
        dones = (position >= self.goal_position) * (velocity >= self.goal_velocity)
        rewards = -np.ones((self._num_envs,))

        self._states = np.stack([position, velocity], axis=-1)
        return np.array(self._states).copy(), rewards, dones, {}

    #def vec_reset(self, num_envs=None):
    #    if num_envs is None:
    #        assert self._num_envs is not None
    #        n = self._num_envs
    #    else:
    #        self._num_envs = num_envs
    #    self._states = np.concatenate([self.np_random.uniform(low=-0.6, high=-0.4, size=(num_envs, 1)),
    #                                   np.ones((num_envs, 1))],
    #                                  axis=-1)
    #    return np.array(self._states).copy()

    def vec_set_state(self, states):
        self._num_envs = len(states)
        self._states = states.copy()

    #def render(self, mode='human', iteration=None):
    #    screen_width = 600
    #    screen_height = 400

    #    world_width = self.max_position - self.min_position
    #    scale = screen_width / world_width
    #    carwidth = 40
    #    carheight = 20

    #    if self.viewer is None:
    #        from gym.envs.classic_control import rendering
    #        self.viewer = rendering.Viewer(screen_width, screen_height)
    #        xs = np.linspace(self.min_position, self.max_position, 100)
    #        ys = self._height(xs)
    #        xys = list(zip((xs - self.min_position) * scale, ys * scale))

    #        self.track = rendering.make_polyline(xys)
    #        self.track.set_linewidth(4)
    #        self.viewer.add_geom(self.track)

    #        clearance = 10

    #        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
    #        car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #        car.add_attr(rendering.Transform(translation=(0, clearance)))
    #        self.cartrans = rendering.Transform()
    #        car.add_attr(self.cartrans)
    #        self.viewer.add_geom(car)
    #        frontwheel = rendering.make_circle(carheight / 2.5)
    #        frontwheel.set_color(.5, .5, .5)
    #        frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
    #        frontwheel.add_attr(self.cartrans)
    #        self.viewer.add_geom(frontwheel)
    #        backwheel = rendering.make_circle(carheight / 2.5)
    #        backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
    #        backwheel.add_attr(self.cartrans)
    #        backwheel.set_color(.5, .5, .5)
    #        self.viewer.add_geom(backwheel)
    #        flagx = (self.goal_position - self.min_position) * scale
    #        flagy1 = self._height(self.goal_position) * scale
    #        flagy2 = flagy1 + 50
    #        flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
    #        self.viewer.add_geom(flagpole)
    #        flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
    #        flag.set_color(.8, .8, 0)
    #        self.viewer.add_geom(flag)

    #    pos = self.state[0]
    #    self.cartrans.set_translation((pos - self.min_position) * scale, self._height(pos) * scale)
    #    self.cartrans.set_rotation(math.cos(3 * pos))

    #    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    #def get_keys_to_action(self):
    #    return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}  # control with left and right arrow keys

    #def close(self):
    #    if self.viewer:
    #        self.viewer.close()
    #        self.viewer = None
