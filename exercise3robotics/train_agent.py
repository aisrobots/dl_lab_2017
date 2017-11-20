import numpy as np
import tensorflow as tf

# custom modules
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this script assumes you did generate your data with the get_data.py script
# you are of course allowed to change it and generate data here but if you
# want this to work out of the box first run get_data.py
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)

# 1. train
######################################
# TODO implement your training here!
# you can get the full data from the transition table like this:
#
# # both train_data and valid_data contain tupes of images and labels
# train_data = trans.get_train()
# valid_data = trans.get_valid()
# 
# alternatively you can get one random mini batch line this
#
# for i in range(number_of_batches):
#     x, y = trans.sample_minibatch()
######################################

# 2. save your trained model


