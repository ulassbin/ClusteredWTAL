# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import os
import sys
import time
import copy
import json
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from utils_cola import AverageMeter
from eval.eval_detection import ANETdetection
from terminaltables import AsciiTable
import utils_cola as utils

class MetricsAdapter:
    def __init__(self):
        pass

    def form_proposals(self, final_res, acc, cfg, label, video_scores, cas, actionness, vid, vid_num_seg):
        print('Vid {}, vidnumseg {}'.format(vid[0], vid_num_seg))
        label_np = label.cpu().data.numpy()
        score_np = video_scores[0].cpu().data.numpy()
        pred_np = np.where(score_np < cfg.CLASS_THRESH, 0, 1)
        correct_pred = np.sum(label_np == pred_np, axis=1)
        acc.update(float(np.sum((correct_pred == cfg.NUM_CLASSES))), correct_pred.shape[0])

        pred = np.where(score_np >= cfg.CLASS_THRESH)[0]  # indexes
        if len(pred) == 0:
          pred = np.array([np.argmax(score_np)])  # If no predictions are made just choose the action with the highest score
        cas_pred = utils.get_pred_activations(cas, pred, cfg)  # filter with video prediction scores?

        aness_pred = utils.get_pred_activations(actionness, pred, cfg)
        proposal_dict = utils.get_proposal_dict(cas_pred, aness_pred, pred, score_np, vid_num_seg, cfg)

        final_proposals = [utils.nms(v, cfg.NMS_THRESH) for _, v in proposal_dict.items()]
        final_res['results'][vid[0]] = utils.result2json(final_proposals, cfg.CLASS_DICT)
        return final_res, acc

    def write_results_to_json(self, final_res, cfg):
        json_path = os.path.join(cfg.OUTPUT_PATH, 'result.json')
        json.dump(final_res, open(json_path, 'w'))
        return json_path

    def getmAp(self, cfg, json_path):
        anet_detection = ANETdetection(cfg.GT_PATH, json_path,
                          subset='test', tiou_thresholds=cfg.TIOU_THRESH,
                          verbose=False, check_status=False)  # compares results on json lvl
        mAP, average_mAP = anet_detection.evaluate()
        return mAP, average_mAP

if __name__ == '__main__':
    print('=> Loaded Adapter')