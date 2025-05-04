import numpy as np
from utils.utils import DiscretizeWrapper


class Discretize(DiscretizeWrapper):
    """
    Discretize class: Discretizes a continous gym environment


    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
        * self.state_points (np.ndarray): grid that contains the real values of the discretization

        * self.obs_n (int): number of discrete points

        * self.transitions (np.ndarray): transition matrix of size (S+1, A, S+1). The last state corresponds to the sink
                                         state
        * self.rewards (np.ndarray): reward matrix of size (S+1, A, S+1). The last state corresponds to the sink state

        * self.get_id_from_coordinates(coordinate_vector) returns the id of the coordinate_vector

        * self.get_state_from_id(id_s): get the continuous state associated to that state id

        * self.get_action_from_id(id_a): get the contiouns action associated to that action id

        * env.set_state(s): resets the environment to the continous state s

        * env.step(a): applies the action a to the environment. Returns next_state, reward, done, env_infos. The
                            last element is not used.
    """

    def get_discrete_state_from_cont_state(self, cont_state):
        """
        Get discrete state from continuous state.
            * self.mode (str): specifies if the discretization is to the nearest-neighbour (nn) or n-linear (linear).

        :param cont_state (np.ndarray): of shape env.observation_space.shape
        :return: A tuple of (states, probs). states is a np.ndarray of shape (1,) if mode=='nn'
                and (2 ^ obs_dim,) if mode=='linear'. probs is the probabability of going to such states,
                it has the same size than states.
        """
        """INSERT YOUR CODE HERE"""
        cont_state = np.expand_dims(cont_state, axis=-1)
        if self.mode == 'nn':
            idx = np.argmin(np.abs(self.state_points - cont_state), axis=1)
            states = self.get_id_from_coordinates(idx)
            states = np.array([states])
            probs = np.array([1])
        elif self.mode == 'linear':
            import itertools
            max_bound = np.max(self.state_points, axis=1)
            min_bound = np.min(self.state_points, axis=1)
            cont_state = np.maximum(np.minimum(cont_state.squeeze(), max_bound), min_bound)
            cont_state = np.expand_dims(cont_state, axis=-1)
            low = np.sum(cont_state >= self.state_points[:, :-1], axis=1) - 1
            high = low + 1
            low_high = np.stack([low, high], axis=1)
            coors = list(itertools.product(*low_high))
            coors = np.stack(coors, axis=1)
            id_s = self.get_id_from_coordinates(coors.T)
            states = np.array(id_s)

            n_dim = self.state_points.shape[0]
            low = self.state_points[np.arange(n_dim), low]
            high = self.state_points[np.arange(n_dim), high]

            left_dist = cont_state.squeeze() - low
            right_dist = high - cont_state.squeeze()
            rl_dist = np.stack([right_dist, left_dist], axis=1)

            weights = list(itertools.product(*rl_dist))
            weights = np.stack(weights, axis=1)
            probs = np.prod(weights, axis=0)
            probs = probs / np.sum(probs)
        else:
            raise NotImplementedError
        return states, probs

    def add_transition(self, id_s, id_a):
        """
        Populates transition and reward matrix (self.transition and self.reward)
        :param id_s (int): discrete index of the the state
        :param id_a (int): discrete index of the the action

        """
        env = self._wrapped_env
        obs_n = self.obs_n

        state = self.get_state_from_id(id_s)
        env.set_state(state)
        next_state, reward, done, _ = env.step(self.get_action_from_id(id_a))
        if done:
            next_id_s = [obs_n]
            probs = [1]
        else:
            next_id_s, probs = self.get_discrete_state_from_cont_state(next_state)
        
        for s, p in zip(next_id_s, probs):
            self.transitions[id_s, id_a, s] = p
            self.rewards[id_s, id_a, s] = reward
        
    def add_done_transitions(self):
        """
        Populates transition and reward matrix for the sink state (self.transition and self.reward). The sink state
        corresponds to the last state (self.obs_n or -1).
        """
        obs_n = self.obs_n
        self.transitions[obs_n, :, obs_n] = 1
        self.rewards[obs_n, :, obs_n] = 0
