import numpy as np
import os
from easydict import EasyDict as edict

cfg = edict()

cfg.MERGING = False
cfg.COLA_UTILS = True
cfg.QUICK_RUN = True
cfg.GPU_ID = '0'
cfg.SELF_LEARN_SCALE = 0.1
cfg.FUSION = 'simple'
cfg.SCORE_CONFIG = [['ScoringFunctions.wide_short_scoring', 'ScoringFunctions.stddev_scoring', 'ScoringFunctions.median_shift_scoring'], [1,1,1]]
cfg.LR = '[0.0001]*10000'
cfg.NUM_ITERS = len(eval(cfg.LR))
cfg.NUM_EPOCHS = 10000
cfg.NUM_CLASSES = 20
cfg.MODAL = 'rgb'
cfg.TEST_FREQ = 5 #25 # Every 5 epoch
cfg.PRINT_FREQ = 5
cfg.FEATS_DIM = 1024
cfg.BATCH_SIZE = 3
cfg.VID_PATH = '/home/ulas/Documents/Datasets/CoLA/data/THUMOS14/features/train/rgb'
cfg.DATA_PATH = '/home/ulas/Documents/Datasets/CoLA/data/THUMOS14'
cfg.CLUSTER_PATH= '/home/ulas/Documents/PhD/2.Codes/clustering/data/data_new/data'
cfg.OUTPUT_PATH= '/home/ulas/Documents/PhD/2.Codes/clustering/results'
cfg.NUM_WORKERS = 8
cfg.LAMBDA = 0.01
cfg.TEMPORAL_LENGTH = 2614
cfg.CLASS_THRESH = 0.5 # Anything over 0.5 is considered to be a prediction for that class
cfg.MIN_PROPOSAL_LENGTH = 5
cfg.CAS_THRESH = np.linspace(0.1, 0.7, 14)
cfg.ANESS_THRESH = np.linspace(0.1, 0.7, 14) if cfg.COLA_UTILS else 0.5
cfg.NMS_THRESH = 0.7
cfg.TIOU_THRESH = np.linspace(0.1, 0.7, 7)
cfg.UP_SCALE = 1 # 24 prev
cfg.GT_PATH = os.path.join(cfg.DATA_PATH, 'gt.json')
cfg.SEED = 0
cfg.FEATS_FPS = 25
cfg.NUM_SEGMENTS = 750
cfg.CLASS_DICT = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 
                  'CleanAndJerk': 3, 'CliffDiving': 4, 'CricketBowling': 5, 
                  'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 
                  'GolfSwing': 9, 'HammerThrow': 10, 'HighJump': 11, 
                  'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 
                  'Shotput': 15, 'SoccerPenalty': 16, 'TennisSwing': 17, 
                  'ThrowDiscus': 18, 'VolleyballSpiking': 19}
