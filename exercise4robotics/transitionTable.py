import numpy as np

class TransitionTable:

    # basic funcs

    def __init__(self, state_siz, act_num, hist_len,
                       minibatch_size, max_transitions):
        self.state_siz = state_siz
        self.act_num = act_num
        self.hist_len  = hist_len
        self.batch_size = minibatch_size
        self.max_transitions = max_transitions

        # memory for state transitions
        self.states  = np.zeros((max_transitions, state_siz*hist_len))
        self.actions = np.zeros((max_transitions, act_num))
        self.next_states = np.zeros((max_transitions, state_siz*hist_len))
        self.rewards = np.zeros((max_transitions, 1))
        self.terminal = np.zeros((max_transitions, 1))
        self.top = 0
        self.bottom = 0
        self.size = 0

    # helper funcs
    def add(self, state, action, next_state, reward, terminal):
        self.states[self.top] = state
        self.actions[self.top] = action
        self.next_states[self.top] = next_state
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal
        if self.size == self.max_transitions:
            self.bottom = (self.bottom + 1) % self.max_transitions
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_transitions

    def one_hot_action(self, actions):
        actions = np.atleast_2d(actions)
        one_hot_actions = np.zeros((actions.shape[0], self.act_num))
        for i in range(len(actions)):
            one_hot_actions[i, int(actions[i])] = 1
        return one_hot_actions

    def sample_minibatch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        state      = np.zeros((self.batch_size, self.state_siz*self.hist_len), dtype=np.float32)
        action     = np.zeros((self.batch_size, self.act_num), dtype=np.float32)
        next_state = np.zeros((self.batch_size, self.state_siz*self.hist_len), dtype=np.float32)
        reward     = np.zeros((self.batch_size, 1), dtype=np.float32)
        terminal   = np.zeros((self.batch_size, 1), dtype=np.float32)
        for i in range(batch_size):
            index = np.random.randint(self.bottom, self.bottom + self.size)
            state[i]         = self.states.take(index, axis=0, mode='wrap')
            action[i]        = self.actions.take(index, axis=0, mode='wrap')
            next_state[i]    = self.next_states.take(index, axis=0, mode='wrap')
            reward[i]        = self.rewards.take(index, axis=0, mode='wrap')
            terminal[i]      = self.terminal.take(index, axis=0, mode='wrap')
        return state, action, next_state, reward, terminal
