import numpy as np
from random import randrange
# custom modules
from utils import State
from maps import maps

class Simulator:

    # basic funcs

    def __init__(self, map_ind, cub_siz, pob_siz, act_num):
        self.map_ind = map_ind
        self.cub_siz = cub_siz
        self.pob_siz = pob_siz
        self.bot_ind = 0 # bot's index in obj_pos
        self.tgt_ind = 1 # bot's index in obj_pos
        self.obs_ind = 2 # bot's index in obj_pos
        self.bot_clr_ind = 2 # blue
        self.tgt_clr_ind = 1 # green
        self.obs_clr_ind = 0 # red
        # state
        self.state_dim = 2
        # action
        self.act_dim   = 2 # 0: ^V; 1: <>
        self.act_num   = act_num # {o, ^, V, <, >}
        self.act_pos_ind = np.array([
            [ 0,  0], # o
            [-1,  0], # ^
            [ 1,  0], # V
            [ 0, -1], # <
            [ 0,  1]  # >
        ])
        self.reset_map(self.map_ind)

    def __del__(self):
        print("Garbage collected.")

    # reset funcs

    def reset_map(self, map_ind):
        self.map     = maps[self.map_ind]
        self.map_hei = self.map.shape[0]
        self.map_wid = self.map.shape[1]
        self.bot_pos_old = np.array([self.map_hei, self.map_wid], int)
        self.tgt_pos_old = np.array([self.map_hei, self.map_wid], int)
        # parse map file
        obs_num = np.sum(self.map)
        self.obj_num = 2 + obs_num  # bot+tgt+#obs
        self.fre_pos = np.ndarray((self.map_hei * self.map_wid - obs_num, self.state_dim), int) # free locations: candidates for bot & tgt
        self.obj_pos = np.ndarray((self.obj_num, self.state_dim), int) # keep track of all objects, including bot tgt & obs
        self.obj_pos[self.bot_ind][0] = self.map_hei # to ease drawing
        self.obj_pos[self.bot_ind][1] = self.map_wid # to ease drawing
        self.obj_pos[self.tgt_ind][0] = self.map_hei # to ease drawing
        self.obj_pos[self.tgt_ind][1] = self.map_wid # to ease drawing
        obj_ind = self.obs_ind
        fre_ind = 0
        for y in range(self.map_hei):
            for x in range(self.map_wid):
                if self.map[y][x] == 1:
                    self.obj_pos[obj_ind][0] = y
                    self.obj_pos[obj_ind][1] = x
                    obj_ind += 1
                else:
                    self.fre_pos[fre_ind][0] = y
                    self.fre_pos[fre_ind][1] = x
                    fre_ind += 1
        self.reset_state()
        self.draw_reset()

    def reset_state(self):
        self.state_action   = 0
        self.state_reward   = 0.
        self.state_screen   = np.zeros((self.map_hei*self.cub_siz, self.map_wid*self.cub_siz, 3), dtype=np.uint8)
        self.state_terminal = False
        self.state_pob      = np.zeros((self.pob_siz*self.cub_siz, self.pob_siz*self.cub_siz, 3), dtype=np.uint8)
        return self.get_state()

    # helper funcs

    def get_cube_from_ind(self, y, x):
        return self.cub_siz*y, self.cub_siz*(y+1), self.cub_siz*x, self.cub_siz*(x+1)

    def get_pob_from_ind(self, y, x):
        pob_edg = self.pob_siz // 2
        return self.cub_siz*(y-pob_edg), self.cub_siz*(y+pob_edg+1), self.cub_siz*(x-pob_edg), self.cub_siz*(x+pob_edg+1)

    def get_h_val(self, active_pose, tgt_y, tgt_x):
        return np.abs(active_pose[0] - tgt_y) + np.abs(active_pose[1] - tgt_x)


    def act(self):
        bot_pos_new = self.obj_pos[self.bot_ind, :] + self.act_pos_ind[self.state_action, :]
        if bot_pos_new[0] == self.obj_pos[self.tgt_ind][0] and bot_pos_new[1] == self.obj_pos[self.tgt_ind][1]: # reaching tgt
            self.state_reward   = 1.
            self.state_terminal = True
            self.obj_pos[self.bot_ind, :] = bot_pos_new
        elif self.map[bot_pos_new[0]][bot_pos_new[1]] == 1: # collision
            self.state_reward   = -1.
            self.state_terminal = False
        else:
            self.state_reward   = -0.04
            self.state_terminal = False
            self.obj_pos[self.bot_ind, :] = bot_pos_new

    def get_state(self):
        return State(self.state_action,
                     self.state_reward,
                     self.state_screen,
                     self.state_terminal,
                     self.state_pob)
    
    # drawing funcs

    def draw_cube(self, y, x, clr_ind, clr_val): # draw on channel clr_ind with clr_val
        y1, y2, x1, x2 = self.get_cube_from_ind(y, x)
        self.state_screen[y1:y2, x1:x2, clr_ind] = clr_val

    def draw_reset(self): # reset background & draw obs
        self.state_screen = np.zeros((self.map_hei*self.cub_siz, self.map_wid*self.cub_siz, 3), dtype=np.uint8)
        for obj_ind in range(self.obs_ind, self.obj_num):
            self.draw_cube(self.obj_pos[obj_ind][0],
                           self.obj_pos[obj_ind][1],
                           self.obs_clr_ind, 255)

    def draw_new(self): # erase old bot tgt & draw new tgt
        # black old bot
        if self.bot_pos_old[0] != self.map_hei and self.bot_pos_old[1] != self.map_wid:
            self.draw_cube(self.bot_pos_old[0],
                           self.bot_pos_old[1],
                           self.bot_clr_ind, 0)
        # black old tgt
        if self.tgt_pos_old[0] != self.map_hei and self.tgt_pos_old[1] != self.map_wid:
            self.draw_cube(self.tgt_pos_old[0],
                           self.tgt_pos_old[1],
                           self.tgt_clr_ind, 0)
        # green new tgt
        self.draw_cube(self.obj_pos[self.tgt_ind][0],
                       self.obj_pos[self.tgt_ind][1],
                       self.tgt_clr_ind, 255)

    def draw_step(self): # erase old bot & draw new bot
        # black old bot
        self.draw_cube(self.bot_pos_old[0],
                       self.bot_pos_old[1],
                       self.bot_clr_ind, 0)
        # blue new bot
        self.draw_cube(self.obj_pos[self.bot_ind][0],
                       self.obj_pos[self.bot_ind][1],
                       self.bot_clr_ind, 255)

    def draw_pob(self): # crop pob from screen
        y1, y2, x1, x2 = self.get_pob_from_ind(self.obj_pos[self.bot_ind][0], self.obj_pos[self.bot_ind][1])
        self.state_pob = self.state_screen[y1:y2, x1:x2, :]

    # interfacing funcs

    def newGame(self, tgt_y, tgt_x):
        # 0. setting up
        if self.obj_pos[self.bot_ind][0] != -1 and self.obj_pos[self.bot_ind][1] != -1:
            self.bot_pos_old[0] = self.obj_pos[self.bot_ind][0]
            self.bot_pos_old[1] = self.obj_pos[self.bot_ind][1]
        if self.obj_pos[self.tgt_ind][0] != -1 and self.obj_pos[self.tgt_ind][1] != -1:
            self.tgt_pos_old[0] = self.obj_pos[self.tgt_ind][0]
            self.tgt_pos_old[1] = self.obj_pos[self.tgt_ind][1]
        # 1. assign tgt position
        if tgt_y != None and tgt_x != None:
            self.obj_pos[self.tgt_ind][0] = tgt_y
            self.obj_pos[self.tgt_ind][1] = tgt_x
        else:
            choose_tgt_ind = randrange(self.fre_pos.shape[0])
            self.obj_pos[self.tgt_ind][0] = self.fre_pos[choose_tgt_ind][0]
            self.obj_pos[self.tgt_ind][1] = self.fre_pos[choose_tgt_ind][1]
        # 2. assign bot position
        choose_bot_ind = randrange(self.fre_pos.shape[0])
        self.obj_pos[self.bot_ind][0] = self.fre_pos[choose_bot_ind][0]
        self.obj_pos[self.bot_ind][1] = self.fre_pos[choose_bot_ind][1]
        # 3. wrap up
        self.draw_new()
        self.tgt_pos_old[0] = self.obj_pos[self.tgt_ind][0]
        self.tgt_pos_old[1] = self.obj_pos[self.tgt_ind][1]
        return self.step(0)

    def step(self, action):
        self.state_action = action
        self.state_reward   = -0.04
        self.state_terminal = False
        self.bot_pos_old[0] = self.obj_pos[self.bot_ind][0]
        self.bot_pos_old[1] = self.obj_pos[self.bot_ind][1]
        self.act()
        self.draw_step()
        self.draw_pob()
        return self.get_state()
