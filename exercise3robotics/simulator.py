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

    # helper funcs

    def get_cube_from_ind(self, y, x):
        return self.cub_siz*y, self.cub_siz*(y+1), self.cub_siz*x, self.cub_siz*(x+1)

    def get_pob_from_ind(self, y, x):
        pob_edg = self.pob_siz // 2
        return self.cub_siz*(y-pob_edg), self.cub_siz*(y+pob_edg+1), self.cub_siz*(x-pob_edg), self.cub_siz*(x+pob_edg+1)

    def get_h_val(self, active_pose, tgt_y, tgt_x):
        return np.abs(active_pose[0] - tgt_y) + np.abs(active_pose[1] - tgt_x)

    def astar(self, bot_y, bot_x, tgt_y, tgt_x):
        # 0. setting up
        self.astar_terminal = False
        open_count = 0
        open_list = {} # (y, x): fVal
        clsd_list = {} # (y, x): fVal
        gVal_list = {} # (y, x): gVal
        hVal_list = {} # (y, x): hVal
        came_from = {} # (neighb_y, neighb_x): ((active_y, active_x), act_ind)

        # 1. push start node into open_list
        active_pose = (bot_y, bot_x)
        gVal_list[active_pose] = 0
        hVal_list[active_pose] = self.get_h_val(active_pose, tgt_y, tgt_x)
        open_list[active_pose] = gVal_list[active_pose] + hVal_list[active_pose]
        open_count += 1

        # 2. expand using A*
        while open_count > 0:
            # 0. sort open_list to pop the entry w/ min f score
            min_fVal = None
            for k, v in open_list.items():
                if min_fVal is None or v < min_fVal:
                    min_pose  = k
                    min_fVal = v
            active_pose = min_pose
            clsd_list[active_pose] = open_list[active_pose] # add it in  clsd_list
            open_list.pop(active_pose, "None")              # pop it out open_list
            open_count -= 1
            # 1. iterate through all its possible successors
            for act_ind in range(self.act_num):
                neighb_pose = self.astar_act(active_pose[0], active_pose[1], tgt_y, tgt_x, act_ind)
                if self.astar_terminal: # have reached tgt, stop searching
                    came_from[neighb_pose] = (active_pose, act_ind)
                    self.astar_retrieve_actions(came_from, bot_y, bot_x, tgt_y, tgt_x)
                    return True
                else:
                    if neighb_pose != active_pose:      # if have actually moved
                        if not neighb_pose in clsd_list:  # if not in clsd_list
                            neighb_g = gVal_list[active_pose] + 1
                            neighb_h = self.get_h_val(neighb_pose, tgt_y, tgt_x)
                            neighb_f = neighb_g + neighb_h
                            if not neighb_pose in open_list or open_list[neighb_pose] >= neighb_f:
                                if not neighb_pose in open_list:
                                    open_count += 1
                                open_list[neighb_pose] = neighb_f
                                gVal_list[neighb_pose] = neighb_g
                                hVal_list[neighb_pose] = neighb_h
                                came_from[neighb_pose] = (active_pose, act_ind)
        return False

    def astar_act(self, bot_y, bot_x, tgt_y, tgt_x, act_ind):
        bot_pos_new = np.ndarray(self.state_dim, int)
        bot_pos_new[0] = bot_y + self.act_pos_ind[act_ind][0]
        bot_pos_new[1] = bot_x + self.act_pos_ind[act_ind][1]
        if bot_pos_new[0] == tgt_y and bot_pos_new[1] == tgt_x: # reaching tgt
            self.astar_terminal = True
            return (bot_pos_new[0], bot_pos_new[1])
        elif self.map[bot_pos_new[0]][bot_pos_new[1]] == 1: # collision
            return (bot_y, bot_x)
        else:
            return (bot_pos_new[0], bot_pos_new[1])

    def astar_retrieve_actions(self, came_from, bot_y, bot_x, tgt_y, tgt_x):
        self.astar_act_lst = []
        tmp_pose = (tgt_y, tgt_x)
        while tmp_pose != (bot_y, bot_x):
            self.astar_act_lst.append(came_from[tmp_pose][1])
            tmp_pose = came_from[tmp_pose][0]

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
        # 3. generate A* actions for this current episode
        self.astar(self.obj_pos[self.bot_ind][0],
                   self.obj_pos[self.bot_ind][1],
                   self.obj_pos[self.tgt_ind][0],
                   self.obj_pos[self.tgt_ind][1])
        # 4. wrap up
        self.draw_new()
        self.tgt_pos_old[0] = self.obj_pos[self.tgt_ind][0]
        self.tgt_pos_old[1] = self.obj_pos[self.tgt_ind][1]
        return self.step(0)

    def step(self, action=None):
        if action is None: # NOTE: when no action is given, will take the A* action
            assert len(self.astar_act_lst) > 0
            self.state_action = self.astar_act_lst.pop()
        else:
            self.state_action = action
        self.state_reward   = -0.04
        self.state_terminal = False
        self.bot_pos_old[0] = self.obj_pos[self.bot_ind][0]
        self.bot_pos_old[1] = self.obj_pos[self.bot_ind][1]
        self.act()
        self.draw_step()
        self.draw_pob()
        return self.get_state()
