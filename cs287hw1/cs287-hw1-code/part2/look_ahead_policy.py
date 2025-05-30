import numpy as np
from gym import spaces


class LookAheadPolicy(object):
    """
    Look ahead policy

    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
    * self.horizon (int): Horizon for the look ahead policy

    * act_dim (int): Dimension of the state space

    * value_fun (TabularValueFun):
                - get_values(states): if states is None returns the values of all the states. Otherwise, it returns the
                                      values of the specified states
    * env (Env):
                - vec_set_state(states): vectorized (multiple environments in parallel) version of reseting the
                environment to a state for a batch of states.
                - vec_step(actions): vectorized (multiple environments in parallel) version of stepping through the
                environment for a batch of actions. Returns the next observations, rewards, dones signals, env infos
                (last not used).
    """
    def __init__(self,
                 env,
                 value_fun,
                 horizon,
                 ):
        self.env = env
        self.discount = env.discount
        self._value_fun = value_fun
        self.horizon = horizon

    def get_action(self, state):
        """
        Get the best action by doing look ahead, covering actions for the specified horizon.
        HINT: use np.meshgrid to compute all the possible action sequences.
        :param state:
        :return: best_action (int)
           """
        assert isinstance(self.env.action_space, spaces.Discrete)
        act_dim = self.env.action_space.n

        actions = np.meshgrid(*([np.arange(act_dim)] * self.horizon))
        actions = np.stack([action.flatten() for action in actions], axis=1).T
        returns = self.get_returns(state, actions)
        best_action = actions[0, np.argmax(returns)]
        return best_action

    def get_returns(self, state, actions):
        """
        :param state: current state of the policy
        :param actions: array of actions of shape [horizon, num_acts]
        :return: returns for the specified horizon + self.discount ^ H value_fun
        HINT: Make sure to take the discounting and done into acount!
        """
        assert self.env.vectorized
        assert self.horizon == actions.shape[0]
        states = np.array([state] * actions.shape[1])
        returns = 0

        for h in range(self.horizon):
            self.env.vec_set_state(states)
            states, rewards, dones, _ = self.env.vec_step(actions[h, :])
            returns += (self.discount**h) * rewards
        returns += (self.discount ** self.horizon) * self._value_fun.get_values(states)
        return returns

    def update(self, actions):
        pass
