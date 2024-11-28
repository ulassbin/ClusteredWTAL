import numpy as np
import torch
import copy
import unittest
import matplotlib.pyplot as plt
import math
import time

def getProposalItemCount(proposals):
  return sum(len(class_proposals) for batch_proposals in proposals for class_proposals in batch_proposals)


def create_proposal(batch_id, class_id, start, end, threshold, cas, index_to_seconds = 16 / 30, borders = 0.2):
  current_length = end - start
  wide_start = max(0, start - int(borders * current_length) - 1)
  wide_end = min(cas.shape[1] - 1, end + int(borders * current_length) + 1)
  #print('Cas shape ', cas.shape)
  #print('S {} e{} - ws{} we{} - max {}'.format(start,end,wide_start,wide_end,cas.shape[1]-1))
  proposal_data = [cas[batch_id, start:end, class_id].cpu().numpy().copy(), cas[batch_id, wide_start:wide_end, class_id].cpu().numpy().copy()]
  return [class_id, start * index_to_seconds, end * index_to_seconds, threshold, proposal_data]


class ScoringFunctions:
  @staticmethod
  def wide_short_scoring(batch_proposals, alpha=0.1): # Scores proposals minus the edges, favoring proposals with sharp edges
    scored_proposals = [[] for _ in range(len(batch_proposals))]
    for i, class_proposals in enumerate(batch_proposals):
      for item in class_proposals:
        if(len(item) != 5):
          continue
        class_id, start, end, threshold, data = item
        base_data = data[0]
        wide_data = data[-1]
        base_score = np.mean(base_data)
        #if(len(base_data) == len(wide_data)):
        #  print('Base data and wide data are same {}  cid {} thresh {}'.format(len(base_data), class_id, threshold))
        edge_score = (np.sum(wide_data) - np.sum(base_data)) / (len(wide_data) - len(base_data) + 1e-6)
        score = base_score - alpha * edge_score
        scored_proposals[i].append([class_id, start, end, threshold, score])
    return scored_proposals

  @staticmethod
  def stddev_scoring(batch_proposals, alpha=1): # Favors consistent proposals over fragmented ones
    scored_proposals = [[] for _ in range(len(batch_proposals))]
    for i, class_proposals in enumerate(batch_proposals):
      for item in class_proposals:
        if len(item) != 5:
          continue
        class_id, start, end, threshold, data = item
        base_data = data[0]
        base_score = np.mean(base_data)
        base_stddev = np.std(base_data)
        score = base_score / (base_stddev + alpha)
        scored_proposals[i].append([class_id, start, end, threshold, score])
    return scored_proposals

  @staticmethod
  def median_shift_scoring(batch_proposals): # Favors proposals that are centered
    scored_proposals = [[] for _ in range(len(batch_proposals))]
    for i, class_proposals in enumerate(batch_proposals):
      for item in class_proposals:
        if(len(item) != 5):
            continue
        class_id, start, end, threshold, data = item
        base_data = data[0]
        wide_data = data[-1]
        base_score = np.mean(base_data)
        time_length = len(base_data)
        sorted_indices = np.argsort(base_data)
        median_base_index = sorted_indices[len(base_data) // 2]
        sorted_indices = np.argsort(wide_data)
        median_wide_index = sorted_indices[len(wide_data) // 2]
        shift = abs(median_base_index - median_wide_index) / (time_length + 1e-6)
        score = -shift # Try to minimize the shift
        scored_proposals[i].append([class_id, start, end, threshold, score])
    return scored_proposals

  @staticmethod
  def combine_scorings(score_list, scoring_weights):
    num_classes = len(score_list[0])
    num_scores = len(score_list)
    final_scores = [[] for _ in range(num_classes)]
    for i in range(num_classes):
      score = 0
      num_proposals = len(score_list[0][i]) # same for any score_list[m][i] = score_list[0][i}
      for j in range(num_proposals):
        score = 0
        for k in range(num_scores):
            score += scoring_weights[k] * score_list[k][i][j][-1]
        final_scores[i].append([score_list[0][i][j][0], score_list[0][i][j][1], score_list[0][i][j][2], score_list[0][i][j][3], score])
    return final_scores

# Cas to proposal routines

class CasToProposals:
  def __init__(self, min_proposal_length, fps, score_config):
    self.min_proposal_length = min_proposal_length
    self.fps = fps
    self.score_config = score_config
    self.index_to_seconds = 16 / fps
    self.borders = 0.2

  def extract_proposals(self, class_cas, time_length, k, i, threshold, cas):
    start, end = -1, -1
    batch_proposal = []
    for j in range(time_length):
      if class_cas[j] == 1:
        if start == -1:
          start = j
        end = j
      else:
        if start != -1 and (end - start) > self.min_proposal_length:
          batch_proposal.append(create_proposal(k, i, start, end, threshold, cas))
        start, end = -1, -1
    if start != -1 and (time_length - 1 - start) > self.min_proposal_length:
      batch_proposal.append(create_proposal(k, i, start, time_length - 1, threshold, cas))
    return batch_proposal
  
#  def create_proposal(self, k, i, start, end, threshold, cas):
#    current_length = end - start
#    wide_start = max(0, start - int(self.borders * current_length) - 1)
#    wide_end = min(cas.shape[1] - 1, end + int(self.borders * current_length) + 1)
#    proposal_data = [cas[k, start:end, i].cpu().numpy().copy(), cas[k, wide_start:wide_end, i].cpu().numpy().copy()]
#    return [i, start * self.index_to_seconds, end * self.index_to_seconds, threshold, proposal_data]

  def cas_to_proposals(self, cas, threshold_list):
    batch = cas.shape[0]
    num_classes = cas.shape[2]
    score_metrics, score_weights = self.score_config
    proposals = [[[] for _ in range(num_classes)] for _ in range(batch)]

    for threshold in threshold_list:
      cas_thresh = cas >= threshold
      for k in range(batch):
        batch_proposal = [[] for _ in range(num_classes)]
        for i in range(num_classes):
          class_cas = cas_thresh[k, :, i]
          time_length = len(class_cas)
          batch_proposal[i].extend(self.extract_proposals(class_cas, time_length, k, i, threshold, cas))
        scores = [metric(batch_proposal) for metric in score_metrics]
        scored_proposals = ScoringFunctions.combine_scorings(scores, score_weights)
        for m, scored_class in enumerate(scored_proposals):
          proposals[k][m].extend(scored_class)
    return proposals


# NMS routines

class NMS:
  def __init__(self, score_config, seconds_to_index, cas, borders=0.2):
    self.score_config = score_config
    self.seconds_to_index = seconds_to_index
    self.cas = cas
    self.borders = borders

  def check_and_merge_proposals(self, first_proposal, second_proposal, batch_id = 0):
    score_metrics, score_weights = self.score_config
    class_id1, start1, end1, threshold1, score1 = first_proposal
    class_id2, start2, end2, threshold2, score2 = second_proposal
    if class_id1 != class_id2:
      if score1 > score2:
        return first_proposal
      return second_proposal
    
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_threshold = min(threshold1, threshold2)
    union_start_index = int(union_start * self.seconds_to_index)
    union_end_index = int(union_end * self.seconds_to_index)
    
    proposal_item = create_proposal(batch_id, class_id1, union_start_index, union_end_index,
                                    union_threshold, self.cas)

    # Calculate the union score
    score_list = []
    for metric in score_metrics:
      score_list.append(metric([[proposal_item]])) # Encapsulating in batch,class structure
    final_scores = ScoringFunctions.combine_scorings(score_list, score_weights)
    union_score = final_scores[-1][-1][-1]
    if union_score > score1 and union_score > score2:  # We got a good merge!
      return [class_id1, union_start, union_end, union_threshold, union_score]
    elif score1 > score2:
      return first_proposal
    return second_proposal

  @staticmethod
  def remove_class_dimension(proposals):
    flat_proposals = [item for sublist in proposals for item in sublist]
    flat_proposals = sorted(flat_proposals, key=lambda x: x[4], reverse=True)
    return flat_proposals

  @staticmethod
  def add_class_dimension(proposals, num_classes):
    classed_proposals = [[] for _ in range(num_classes)]
    for proposal in proposals:
      class_id = proposal[0]
      classed_proposals[class_id].append(proposal)
    return classed_proposals

  def nms(self, proposals, nms_threshold, merging=False):
    batch_size = len(proposals)
    num_classes = len(proposals[0])
    final_proposals = [[[] for _ in range(num_classes)] for _ in range(batch_size)]
    for batch_id in range(batch_size):
      batch_proposals = proposals[batch_id]
      flattened_proposals = self.remove_class_dimension(batch_proposals)
      batch_final_proposals = []
      for i in range(len(flattened_proposals)):
        current_proposal = flattened_proposals[i]
        if len(batch_final_proposals) == 0:
          batch_final_proposals.append(current_proposal)
        else:
          overlap = False
          for j in range(len(batch_final_proposals)):
            start = max(current_proposal[1], batch_final_proposals[j][1])
            end = min(current_proposal[2], batch_final_proposals[j][2])
            intersection = max(0, end - start)
            union = (current_proposal[2] - current_proposal[1]) + (batch_final_proposals[j][2] - batch_final_proposals[j][1]) - intersection
            iou = intersection / union
            if iou > nms_threshold:
              overlap = True
              if merging:
                batch_final_proposals[j] = self.check_and_merge_proposals(current_proposal, batch_final_proposals[j], batch_id)
                break
          if not overlap:
            batch_final_proposals.append(current_proposal)
      # Convert to classwise separated container
      final_proposals[batch_id] = self.add_class_dimension(batch_final_proposals, num_classes)
    return final_proposals


# filterings

def actionness_filter_proposals(proposals, actionness, cfg):
  # Proposals are in the form of proposal[batch][num_class] = [class, start, end, threshold,score]
  num_batch = actionness.shape[0]
  num_classes = cfg.NUM_CLASSES
  seconds_to_index = cfg.FEATS_FPS / 16
  threshold = cfg.ANESS_THRESH
  filtered_proposals = [ [[] for _ in range(num_classes)] for _ in range(num_batch)]
  assert num_batch == len(proposals), "Number of proposals and actionness should match {} to {}".format(num_batch, len(proposals))
  assert num_classes == len(proposals[0]), "Number of classes in proposals and config should match {} to {}".format(num_classes,len(proposals[0]))
  for batch_id in range(num_batch):
    batch_proposal = proposals[batch_id]
    batch_actionness = actionness[batch_id]
    for class_id in range(len(batch_proposal)):
      class_prop = batch_proposal[class_id]
      for proposal in class_prop:
        start_frame = int(proposal[1] * seconds_to_index)
        end_frame = int(proposal[2] * seconds_to_index)
        actionness_values = batch_actionness[start_frame:end_frame]
        if np.mean(actionness_values.cpu().numpy()) > threshold:
          filtered_proposals[batch_id][class_id].append(proposal)
  return filtered_proposals

class ProposalGenie:
  def __init__(self, cfg, score_config):
    self.cfg = cfg
    if not isinstance(score_config, list):
      raise ValueError('Score config should be a list')
    self.score_config = score_config
    # Support for function + string input
    for i, item in enumerate(self.score_config[0]):
      if isinstance(item, str): # String support
        self.score_config[0][i] = eval(item)
    self.min_proposal_length = cfg.MIN_PROPOSAL_LENGTH
    self.fps = cfg.FEATS_FPS
    self.nms_threshold = cfg.NMS_THRESH
    self.aness_threshold = cfg.ANESS_THRESH
    self.num_classes = cfg.NUM_CLASSES
    self.seconds_to_index = self.fps / 16


  def cas_to_proposals(self, cas):
    proposals = CasToProposals(self.min_proposal_length, self.fps, self.score_config).cas_to_proposals(cas, self.cfg.CAS_THRESH)
    return proposals

  def filter_proposals(self, proposals, actionness):
    filtered_proposals = actionness_filter_proposals(proposals, actionness, self.cfg)
    return filtered_proposals

  def nms(self, cas, proposals, merging=True):
    return NMS(self.score_config, self.seconds_to_index, cas).nms(proposals, self.nms_threshold, merging)

  def generate_proposals(self, cas, actionness, merging=True):
    proposals = self.cas_to_proposals(cas)
    filtered_proposals = self.filter_proposals(proposals, actionness)
    nms_proposals = self.nms(cas, filtered_proposals, merging)
    return nms_proposals


  @staticmethod
  def visualize(cas, proposals, fps, clr='r'):
    index_to_seconds = 16 / fps
    batch_size = cas.shape[0]
    num_classes = cas.shape[2]
    
    # Compute grid dimensions
    cols = math.ceil(math.sqrt(num_classes))  # Number of columns
    rows = math.ceil(num_classes / cols)     # Number of rows

    for batch_id in range(batch_size):
      fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
      axes = axes.flatten()  # Flatten axes array for easier indexing
      time_axis = torch.arange(cas[batch_id].shape[0]) * index_to_seconds
      for class_id in range(num_classes):
        ax = axes[class_id]
        # Plot CAS values for the class
        ax.plot(time_axis, cas[batch_id, :, class_id].cpu().numpy(), label=f'Class {class_id}')
        
        # Add proposals as shaded regions
        batch_proposals = proposals[batch_id]
        proposals_class = batch_proposals[class_id]
        for proposal in proposals_class:
          start = proposal[1]  # Time in seconds
          end = proposal[2]    # Time in seconds
          threshold = proposal[3]
          ax.axvspan(start, end, color=clr, alpha=0.3, label='Proposal T: {:2g} S:{:.2g}'.format(threshold, proposal[-1]))
        ax.set_title(f'Class {class_id}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('CAS Value')
        ax.legend()
        # Hide unused subplots if any
        for idx in range(num_classes, len(axes)):
          fig.delaxes(axes[idx])

      fig.suptitle(f'Batch {batch_id}', fontsize=16)
      plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for suptitle
    plt.show()
    return




class TestProposalGenie(unittest.TestCase):
  def test_generate_proposals(self):
    # Mock inputs
    num_classes = 9
    cas = torch.rand((3, 100, num_classes)) # Example tensor with shape (batch, temporal_length, num_classes)
    actionness = torch.sum(cas, dim=2)
    threshold_list = np.arange(0.1, 0.8, 0.1)
    min_proposal_length = 5
    fps = 30
    score_simple = [[ScoringFunctions.wide_short_scoring], [1]]
    nms_threshold = 0.5
    aness_threshold = 0.5
    cfg = type('Config', (object,), {})
    nms_threshold = 0.5
    aness_threshold = 0.5
    cfg.NUM_CLASSES = num_classes
    cfg.FEATS_FPS = fps
    cfg.NMS_THRESH = nms_threshold
    cfg.ANESS_THRESH = aness_threshold
    cfg.MIN_PROPOSAL_LENGTH = min_proposal_length
    cfg.CAS_THRESH = threshold_list

    # Initialize ProposalGenie
    genie = ProposalGenie(cfg, score_simple)
    # Generate proposals
    proposals = genie.generate_proposals(cas, actionness)
    num_props = getProposalItemCount(proposals)
    print('Num Proposals {}'.format(num_props))
    # Assertions
    self.assertIsInstance(proposals, list)  # Ensure output is a list
    self.assertEqual(len(proposals), cas.shape[0])  # Ensure batch size matches

  def test_filter_proposals(self):
    # Mock inputs
    cfg = type('Config', (object,), {
      'NUM_CLASSES': 9,
      'FEATS_FPS': 30,
      'NMS_THRESH': 0.5,
      'ANESS_THRESH': 0.5,
      'MIN_PROPOSAL_LENGTH': 5,
      'CAS_THRESH': np.arange(0.1, 0.8, 0.1)
    })
    cas = torch.rand((3, 100, cfg.NUM_CLASSES)) # Example tensor with shape (batch, temporal_length, num_classes)
    actionness = torch.sum(cas, dim=2)
    score_simple = [[ScoringFunctions.wide_short_scoring], [1]]

    # Initialize ProposalGenie
    genie = ProposalGenie(cfg, score_simple)
    # Calling each method separately
    proposals = genie.cas_to_proposals(cas)
    filtered_proposals = genie.filter_proposals(proposals, actionness)
    nms_proposals = genie.nms(cas, filtered_proposals)

    num_props = getProposalItemCount(proposals)
    num_filtered_props = getProposalItemCount(filtered_proposals)
    num_nms_props = getProposalItemCount(nms_proposals)
    print('Num Proposals {}, filtered {}, nms {}'.format(num_props, num_filtered_props, num_nms_props))
    genie.visualize(cas, proposals, cfg.FEATS_FPS)
    genie.visualize(cas, filtered_proposals, cfg.FEATS_FPS, 'b')
    genie.visualize(cas, nms_proposals, cfg.FEATS_FPS, 'g')
    
    # Assertions
    self.assertIsInstance(filtered_proposals, list)  # Ensure output is a list
    self.assertEqual(len(filtered_proposals), cas.shape[0])  # Ensure batch size matches
    self.assertGreaterEqual(num_props, num_filtered_props)  # Ensure proposals are filtered

  def test_score_timing(self):
    cas = torch.rand((20, 100, 9))
    actionness = torch.sum(cas, dim=2)
    threshold_list = [0.25, 0.5, 0.75]
    min_proposal_length = 5
    fps = 30
    score_wide_short = [[ScoringFunctions.wide_short_scoring], [1]]
    score_stddev = [[ScoringFunctions.stddev_scoring], [1]]
    score_median_shift = [[ScoringFunctions.median_shift_scoring], [1]]
    score_all = [[ScoringFunctions.wide_short_scoring, ScoringFunctions.stddev_scoring, ScoringFunctions.median_shift_scoring], [1, 1, 1]]

    num_classes = 9
    nms_threshold = 0.5
    aness_threshold = 0.5
    cfg = type('Config', (object,), {})
    cfg.NUM_CLASSES = num_classes
    cfg.FEATS_FPS = fps
    cfg.NMS_THRESH = nms_threshold
    cfg.ANESS_THRESH = aness_threshold
    cfg.MIN_PROPOSAL_LENGTH = min_proposal_length
    cfg.CAS_THRESH = threshold_list

    start = time.time()
    proposals = ProposalGenie(cfg, score_wide_short).generate_proposals(cas, actionness)
    end = time.time()
    print('Time taken for wide short scoring is ', end - start)
    start = time.time()
    proposals = ProposalGenie(cfg, score_stddev).generate_proposals(cas, actionness)
    end = time.time()
    print('Time taken for stddev scoring is ', end - start)
    start = time.time()
    proposals = ProposalGenie(cfg, score_median_shift).generate_proposals(cas, actionness)
    end = time.time()
    print('Time taken for median shift scoring is ', end - start)
    start = time.time()
    proposals = ProposalGenie(cfg, score_all).generate_proposals(cas, actionness)
    end = time.time()
    print('Time taken for all scoring is ', end - start)
    self.assertTrue(True)
  
  def test_merge_timing(self):
    import time
    cas = torch.rand((20, 100, 9))
    actionness = torch.sum(cas, dim=2)
    threshold_list = [0.25, 0.5, 0.75]
    min_proposal_length = 5
    fps = 30
    num_classes = 9
    nms_threshold = 0.5
    aness_threshold = 0.5
    score_all = [[ScoringFunctions.wide_short_scoring, ScoringFunctions.stddev_scoring, ScoringFunctions.median_shift_scoring], [1, 1, 1]]
    cfg = type('Config', (object,), {})
    cfg.NUM_CLASSES = num_classes
    cfg.FEATS_FPS = fps
    cfg.NMS_THRESH = nms_threshold
    cfg.ANESS_THRESH = aness_threshold
    cfg.MIN_PROPOSAL_LENGTH = min_proposal_length
    cfg.CAS_THRESH = threshold_list

    genie = ProposalGenie(cfg, score_all)
    proposals = genie.cas_to_proposals(cas)
    proposals = genie.filter_proposals(proposals, actionness)
    start = time.time()
    filtered_proposals = NMS(score_all, fps / 16, cas).nms(proposals, 0.5, merging=False)
    end = time.time()
    print('Time taken for nms without merging is ', end - start)
    start = time.time()
    filtered_proposals = NMS(score_all, fps / 16, cas).nms(proposals, 0.5, merging=True)
    end = time.time()
    print('Time taken for nms with merging is ', end - start)
    self.assertTrue(True)


if __name__ == '__main__':
  unittest.main()
