import numpy as np

class Options:
    #
    disp_on = True # you might want to set it to False for speed
    map_ind = 1
    change_tgt = False
    states_fil = "states.csv"
    labels_fil = "labels.csv"
    network_fil = "network.json"
    weights_fil = "network.h5"
    # simulator config
    disp_interval = .005
    if map_ind == 0:
        cub_siz = 5
        pob_siz = 5 # for partial observation
        # this defines the goal positionw
        tgt_y = 12
        tgt_x = 11
        early_stop = 50
    elif map_ind == 1:
        cub_siz = 10
        pob_siz = 3 # for partial observation
        # this defines the goal positionw
        tgt_y = 5
        tgt_x = 5
        early_stop = 75
    state_siz = (pob_siz * cub_siz) ** 2 # when use pob as input
    if change_tgt:
        tgt_y = None
        tgt_x = None
    act_num = 5

    # traing hyper params    
    hist_len = 4
    minibatch_size  = 32
    eval_nepisodes  = 10

class State: # return tuples made easy
    def __init__(self, action, reward, screen, terminal, pob):
        self.action   = action
        self.reward   = reward
        self.screen   = screen
        self.terminal = terminal
        self.pob      = pob


# The following functions were taken from scikit-image
# https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py
        
def rgb2gray(rgb):
    if rgb.ndim == 2:
        return np.ascontiguousarray(rgb)

    gray = 0.2125 * rgb[..., 0]
    gray[:] += 0.7154 * rgb[..., 1]
    gray[:] += 0.0721 * rgb[..., 2]

    return gray
