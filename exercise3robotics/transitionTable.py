import numpy as np

class TransitionTable:

    # basic funcs

    def __init__(self, state_siz, act_num, hist_len,
                       minibatch_size, valid_size,
                       states_fil, labels_fil):
        self.state_siz = state_siz
        self.act_num = act_num
        self.hist_len  = hist_len
        self.minibatch_size = minibatch_size
        self.valid_size = valid_size
        self.states_fil = states_fil
        self.labels_fil = labels_fil
        self.minibatchInd = None
        self.load_data()
        self.recent_states = np.zeros([self.hist_len, self.state_siz])

    def __del__(self):
        print("Garbage collected.")

    # helper funcs

    def one_hot(self, labels):
        classes = np.unique(labels)
        n_classes = classes.size
        one_hot_labels = np.zeros(labels.shape + (n_classes,))
        for c in classes:
            one_hot_labels[labels == c, c] = 1
        return one_hot_labels

    def stack_hist(self):
        # stack history
        old_states = self.states.copy()
        self.states = np.zeros([self.size, self.hist_len * self.state_siz])
        full_state = np.zeros([self.hist_len, self.state_siz])
        # use this for indicating the beginning of each of episode (the 1st frame is always w/ label 0)
        old_labels = self.labels.copy()
        # NOTE: the label here is the action that gets the agent into the current state
        # NOTE: but we need the label to be the action the agent should take from this current state
        # NOTE: so we move all the labels up by 1 index, and add a 0 for the last frame
        self.labels = np.append(self.labels, [0], 0)
        self.labels = np.delete(self.labels, 0, 0)
        # labels into one hot encoding
        self.labels = self.one_hot(self.labels)

        for i in range(self.size):
            state = old_states[i, :]
            label = old_labels[i]
            if label == 0:  # starting of a new episode
                full_state = np.zeros([self.hist_len, self.state_siz]) # erase current history
                for j in range(self.hist_len):
                    full_state[j] = state
            else:
                full_state = np.append(full_state, state.reshape(1, self.state_siz), 0)
                if full_state.shape[0] > self.hist_len: # remove the oldest history
                    full_state = np.delete(full_state, 0, 0)
            self.states[i] = full_state.reshape(self.hist_len * self.state_siz)

    def load_data(self):
        self.states = np.loadtxt(self.states_fil, delimiter=',')
        self.labels = np.loadtxt(self.labels_fil, delimiter=',').astype("int")
        assert self.states.shape[0] == self.labels.shape[0]
        self.size   = self.states.shape[0]
        self.minibatchNum = int(self.size - self.valid_size) / int(self.minibatch_size)
        self.minibatchInd = None
        self.stack_hist()
        print("states & labels loaded.")
        print("states stacked w/ history of", self.hist_len)
        self.split_train_valid()
        print("train & valid data splited.")

    def split_train_valid(self):
        train_size = self.size - self.valid_size
        shuffled_ind = np.random.permutation(self.size)
        self.train_states = self.states[shuffled_ind[0:train_size], :].copy()
        self.train_labels = self.labels[shuffled_ind[0:train_size], :].copy()
        self.valid_states = self.states[shuffled_ind[train_size:self.size], :].copy()
        self.valid_labels = self.labels[shuffled_ind[train_size:self.size], :].copy()

    # core funcs

    def add_recent(self, epi_step, curr_state):
        if epi_step == 0: # starting of a new episode
            self.recent_states = np.zeros([self.hist_len, self.state_siz]) # erase current history
            for i in range(self.hist_len):
                self.recent_states[i] = curr_state
        else:
            self.recent_states = np.append(self.recent_states, curr_state.reshape(1, self.state_siz), 0)
            if self.recent_states.shape[0] > self.hist_len: # remove the oldest history
                self.recent_states = np.delete(self.recent_states, 0, 0)

    def get_recent(self):
        return self.recent_states.copy().reshape(1, self.hist_len * self.state_siz)

    # to be used for NeuralPlanner.learn()
    def get_train(self):
        return self.train_states.copy(), self.train_labels.copy()

    # to be used for NeuralPlanner.learn()
    def get_valid(self):
        return self.valid_states.copy(), self.valid_labels.copy()

    # to be used for NeuralPlanner.learn_minibatch()
    def sample_minibatch(self, batch_size=None): # when called w/o args, get minibatch_size using fixed shuffled indices
        if batch_size is None:                   # currently called w/ args is not implemented
            if self.minibatchInd is None:
                self.minibatchInd = 0
            else:
                self.minibatchInd = (self.minibatchInd + 1) % self.minibatchNum
        if self.minibatchInd == 0: # ran through an epoch, now get another shuffled ind for next epoch
            self.minibatchOrder = np.random.permutation(self.size - self.valid_size)
        current_ind = self.minibatchOrder[self.minibatchInd*self.minibatch_size : (self.minibatchInd+1)*self.minibatch_size]
        return self.train_states[current_ind, :].copy(), self.train_labels[current_ind, :].copy()
