import numpy as np
import torch
import copy
import unittest


def getProposalItemCount(proposals):
    count = 0
    for batch_id in range(len(proposals)):
        for class_id in range(len(proposals[batch_id])):
            for proposal in proposals[batch_id][class_id]:
                count += 1
    return count



def interpolate_cas(cas, scale_factor):
    return cas # pass for now

def wide_short_scoring(batch_proposals, alpha=0.1):
    scored_proposals = [[] for _ in range(len(batch_proposals))]
    for i, prop in enumerate(batch_proposals):
        for j, item in enumerate(prop):
            if(len(item) != 5):
                continue
            class_id, start, end, threshold, data = item
            base_data = data[0]
            wide_data = data[-1]
            base_score = np.mean(base_data)
            edge_score = (np.sum(wide_data) - np.sum(base_data)) / (len(wide_data) - len(base_data) + 1e-6)
            score = base_score - alpha * edge_score # This will score the proposal, minus the edges. Making sharp edges less likely to be selected.
            scored_proposals[i].append([class_id, start, end, threshold, score])
    return scored_proposals

def stddev_scoring(batch_proposals, alpha=1): # Favors consistent proposals over fragmented ones
    scored_proposals = [[] for _ in range(len(batch_proposals))]
    for i, prop in enumerate(batch_proposals):
        for j, item in enumerate(prop):
            if(len(item) != 5):
                continue
            class_id, start, end, threshold, data = item
            base_data = data[0]
            base_score = np.mean(base_data)
            base_stddev = np.std(base_data)
            score = base_score / (base_stddev + alpha) # Smoother penalties when using 1 rather than using epsilon
            scored_proposals[i].append([class_id, start, end, threshold, score])
    return scored_proposals

def median_shift_scoring(batch_proposals): # Favors proposals that are centered
    scored_proposals = [[] for _ in range(len(batch_proposals))]
    for i, prop in enumerate(batch_proposals):
        for j, item in enumerate(prop):
            if(len(item) != 5):
                continue
            class_id, start, end, threshold, data = item
            base_data = data[0]
            wide_data = data[-1]
            base_score = np.mean(base_data)
            time_length = len(base_data)
            #median_base = np.median(base_data) # causes floating point issues
            #print(' Base data shape ', base_data.shape)
            #print(' median base val', median_base)
            #print(' base_data ', base_data)
            #print(' Where output ', np.where(base_data == median_base))
            #median_base_index = np.where(base_data == median_base)[0][0]
            sorted_indices = np.argsort(base_data)
            median_base_index = sorted_indices[len(base_data) // 2]
            #median_wide = np.median(wide_data)
            #median_wide_index = np.where(wide_data == median_wide)[0][0]
            sorted_indices = np.argsort(wide_data)
            median_wide_index = sorted_indices[len(wide_data) // 2]
            shift = abs(median_base_index - median_wide_index) / (time_length + 1e-6)
            score = -shift # Try to minimize the shift
            scored_proposals[i].append([class_id, start, end, threshold, score])
    return scored_proposals


config = [[wide_short_scoring, stddev_scoring, median_shift_scoring], [1, 1, 1]]


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

def cas_to_proposals(cas, threshold_list, min_proposal_length, fps, score_config):
    batch = cas.shape[0]
    num_classes = cas.shape[2]
    index_to_seconds = 16 / fps
    borders = 0.2
    score_metrics, score_weights = score_config
    proposals = [[] for _ in range(batch)]
    for threshold in threshold_list:
        cas_thresh = cas >= threshold
        num_positives = torch.sum(cas_thresh, dim=1)
        #print("Number of positives for threshold {} is {}".format(threshold, num_positives))
        # now somehow we should convert cas_thresh into a proposal list of form [class, start, end, score, normalized_score]
        # Lets start with a simple approach
        for k in range(batch):
            batch_proposal = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                class_cas = cas_thresh[k,:,i]
                #print('Cas values for batch {} class {} are {}'.format(k, i, np.sum(cas[k,:,i].cpu().numpy())))
                time_length = len(class_cas)
                start = -1
                end = -1
                for j in range(time_length):
                    if class_cas[j] == 1:
                        if start == -1:
                            start = j
                        end = j
                    else:
                        if start != -1: # We already started a proposal
                            if (end - start) > min_proposal_length:
                                current_length = end - start
                                wide_start = max(0, start - int(borders * current_length) - 1) # factor*length + index_to_seconds offset
                                wide_end = min(time_length-1, end + int(borders * current_length) + 1)
                                #print('Short start {} end {} wide start {} wide end {}'.format(start, end, wide_start, wide_end))
                                proposal_data = [cas[k,start:end,i].cpu().numpy().copy(), cas[k,wide_start:wide_end,i].cpu().numpy().copy()]
                                batch_proposal[i].append([i, start*index_to_seconds, end*index_to_seconds, threshold, copy.deepcopy(proposal_data)])
                                #print("Adding Proposal for class {} start {} end {}".format(i, start*index_to_seconds, end*index_to_seconds))
                            # reset start position
                            start = -1
                            end = -1
                if start != -1: # Finalize the proposal if it continues until the end of class_cas
                    if (time_length - 1 - start) > min_proposal_length: # +-1 index error is possible her (Check later.)
                        current_length = time_length - 1 - start
                        wide_start = max(0, start - int(borders * current_length)- 1)
                        wide_end = min(time_length-1, end + int(borders * current_length) + 1)
                        proposal_data = [cas[k,start:end,i].cpu().numpy().copy(), cas[k,wide_start:wide_end,i].cpu().numpy().copy()]
                        batch_proposal[i].append([i, start*index_to_seconds, end*index_to_seconds, threshold, copy.deepcopy(proposal_data)])
                        #print("Adding Proposal for class {} start {} end {}".format(i, start*index_to_seconds, end*index_to_seconds))
            # Now lets score the batch_proposals
            scored_proposals = [[] for _ in range(num_classes)]
            scores = []
            #print('Batch proposals are ', np.sum([len(proposal) for proposal in batch_proposal]))
            for j in range(len(score_metrics)):
                item = score_metrics[j](batch_proposal)
                scores.append(score_metrics[j](batch_proposal))
            #print('Scores are', scores)
            scored_proposals = combine_scorings(scores, score_weights)
            #print("Scored proposals len ", [np.sum(len(it)) for it in scored_proposals])
            proposals[k] += scored_proposals # Assign computed proposal for batch k!
            #print("Prop batch {}, classes {}".format(len(proposals), len(proposals[0])))
            #print("Batch class {}, scored {}".format(len(batch_proposal), len(scored_proposals)))
    #print("Prop batch {}, classes {}".format(len(proposals), len(proposals[0])))
    return proposals # Proposals is a list with batch, num_classes, proposals


def checkMerges(proposal1, proposal2, score_config, seconds_to_index, cas_batch):
    # Check if the proposals can be merged, if so 
    # calculate the merged score, and if it is better than the individual scores
    # return the merged proposal, otherwise return the original proposal with the highest score
    # proposal is [class_id, start, end, score]
    class_id1, start1, end1, threshold1, score1 = proposal1
    class_id2, start2, end2, threshold2, score2 = proposal2
    # Check if the proposals are of the same class
    temporal_length = cas_batch.shape[0]
    if class_id1 != class_id2:
        if score1 > score2:
                return proposal1
        return proposal2
    cas_class = cas_batch[:, class_id1]
    intersection = max(0, min(end1, end2) - max(start1, start2)) 
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_start_index = int(union_start * seconds_to_index)
    union_end_index = int(union_end * seconds_to_index)
    borders = 0.2
    len_union = union_end - union_start
    union_start_index_wide = max(0, union_start_index - int(borders * len_union) - 1) # correct
    union_end_index_wide = min(temporal_length-1, union_end_index + int(borders * len_union) + 1) # correct
    # Calculate the union score
    score_metrics, score_weights = score_config
    union_score = 0
    data = [cas_batch[union_start_index:union_end_index,class_id1].cpu().numpy(), cas_batch[union_start_index_wide:union_end_index_wide,class_id1].cpu().numpy()]
    union_threshold = min(threshold1, threshold2)
    union_item = [[[class_id1, union_start, union_end, union_threshold, data]]]
    score_list = []
    for i in range(len(score_metrics)):
        score_list.append(score_metrics[i](union_item))
    final_scores = combine_scorings(score_list, score_weights)
    union_score = final_scores[-1][-1][-1]
    if union_score > score1 and union_score > score2: # We got a good merge!
        return [class_id1, union_start, union_end, union_threshold, union_score]
    elif score1 > score2:
        return proposal1
    else:
        return proposal2

def nms(proposals, nms_threshold, score_config, seconds_to_index, cas , merging = True):
    # Sort the proposals based on the normalized score
    # Proposals look like proposal[example_id][class_id] = [class, start, end, score]
    batch_size = len(proposals)
    num_classes = len(proposals[0])
    # Collapse by batch by batch
    final_proposals = [[[] for _ in range(num_classes)] for _ in range(batch_size)]
    for batch_id in range(batch_size):
        batch_proposals = proposals[batch_id]
        collapsed_proposals = [item for sublist in batch_proposals for item in sublist] # Collapsing classes for nms purposes
        collapsed_proposals = sorted(collapsed_proposals, key=lambda x: x[3], reverse=True)
        batch_final_proposals = []
        batch_classed_final_proposals = [[] for _ in range(num_classes)]
        for i in range(len(collapsed_proposals)):
            current_proposal = collapsed_proposals[i]
            if len(batch_final_proposals) == 0:
                batch_final_proposals.append(current_proposal)
            else:
                # Check for overlap with the previous proposals
                overlap = False
                for j in range(len(batch_final_proposals)):
                    # Calculate intersection over union (IoU)
                    start = max(current_proposal[1], batch_final_proposals[j][1])
                    end = min(current_proposal[2], batch_final_proposals[j][2])
                    intersection = max(0, end - start)
                    union = (current_proposal[2] - current_proposal[1]) + (batch_final_proposals[j][2] - batch_final_proposals[j][1]) - intersection
                    iou = intersection / union
                    if iou > nms_threshold:
                        overlap = True
                        if(merging): # maybe we can also have a merging threshold
                            batch_final_proposals[j] = checkMerges(current_proposal, batch_final_proposals[j], score_config, seconds_to_index, cas[batch_id])
                        break
                if not overlap:
                    batch_final_proposals.append(current_proposal) 
        for proposal in batch_final_proposals:
            batch_classed_final_proposals[proposal[0]].append(proposal)
        # Finally append batch_classed_final_proposals to the final list
        final_proposals[batch_id] = copy.deepcopy(batch_classed_final_proposals) # deepcopy to avoid reference issues 
    return final_proposals


# filterings

def actionness_filter_proposals(proposals, actionness, cfg):
    # Proposals are in the form of proposal[batch][num_class] = [class, start, end, score]
    num_batch = actionness.shape[0]
    num_classes = cfg.NUM_CLASSES
    seconds_to_index = cfg.FEATS_FPS / 16
    threshold = cfg.ANESS_THRESH
    filtered_proposals = [ [[] for _ in range(num_classes)] for _ in range(num_batch)]
    assert num_batch == len(proposals), "Number of proposals and actionness should match"
    assert num_classes == len(proposals[0]), "Number of classes in proposals and config should match"
    #print('Num batches {}, num_classes {}, num_proposals {}'.format(len(proposals), [len(proposals[i]) for i in range(len(proposals))], [len(proposals[0][i]) for i in range(len(proposals[0]))]))
    for batch_id in range(num_batch):
        batch_proposal = proposals[batch_id]
        batch_actionness = actionness[batch_id]
        for class_id in range(len(batch_proposal)):
            #print("Len of batch proposal ", len(batch_proposal))
            #print("Batch_prop class ", (batch_proposal[class_id]))
            for proposal in batch_proposal[class_id]:
                #print("Prop ", proposal)
                start_frame = int(proposal[1] * seconds_to_index)
                end_frame = int(proposal[2] * seconds_to_index)
                #print(" Len {}, start {}, end {}".format(batch_actionness.shape, start_frame, end_frame))
                actionness_values = batch_actionness[start_frame:end_frame]
                if np.mean(actionness_values.cpu().numpy()) > threshold:
                    filtered_proposals[batch_id][class_id].append(proposal)
    return filtered_proposals


# Write a function to visualize cas values and proposals 
def visualize(cas, proposals, fps, clr='r'):
    import matplotlib.pyplot as plt
    import math
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

        # Wait for the plot window to be closed
    return


# define a proposal function test here
class TestCasToProposals(unittest.TestCase):
    def test_cas_to_proposals(self):
        # Mock inputs
        cas = torch.rand((20, 100, 9)) # Example tensor with shape (batch, temporal_length, num_classes)
        threshold_list = [0.25, 0.5, 0.75]
        min_proposal_length = 5
        fps = 30
        seconds_to_index = fps / 16
        #score = [[wide_short_scoring, stddev_scoring, median_shift_scoring], [1, 1, 1]]
        score_simple = [[wide_short_scoring], [1]]
        # Call the function
        proposals = cas_to_proposals(cas, threshold_list, min_proposal_length, fps, score_simple)
        nms_threshold = 0.5
        filtered_proposals = nms(proposals, nms_threshold, score_simple, seconds_to_index, cas , False)
        num_props = getProposalItemCount(proposals)
        num_filtered_props = getProposalItemCount(filtered_proposals)
        print('Num Proposals {}, filtered {}'.format(getProposalItemCount(proposals), getProposalItemCount(filtered_proposals)))
        #visualize(cas, proposals, fps)
        #visualize(cas, filtered_proposals, fps, 'g')
        # Assertions
        self.assertIsInstance(proposals, list)  # Ensure output is a list
        self.assertEqual(len(proposals), cas.shape[0])  # Ensure batch size matches
        self.assertGreaterEqual(num_props, num_filtered_props)  # Ensure proposals are generated

    # Lets also adda a test where we check timings of several scoring function sets!
    def test_score_timing(self):
        cas = torch.rand((20, 100, 9))
        threshold_list = [0.25, 0.5, 0.75]
        min_proposal_length = 5
        fps = 30
        seconds_to_index = fps / 16
        score_wide_short = [[wide_short_scoring], [1]]
        score_stddev = [[stddev_scoring], [1]]
        score_median_shift = [[median_shift_scoring], [1]]
        score_all = [[wide_short_scoring, stddev_scoring, median_shift_scoring], [1, 1, 1]]
        import time
        start = time.time()
        proposals = cas_to_proposals(cas, threshold_list, min_proposal_length, fps, score_wide_short)
        end = time.time()
        print('Time taken for wide short scoring is ', end - start)
        start = time.time()
        proposals = cas_to_proposals(cas, threshold_list, min_proposal_length, fps, score_stddev)
        end = time.time()
        print('Time taken for stddev scoring is ', end - start)
        start = time.time()
        proposals = cas_to_proposals(cas, threshold_list, min_proposal_length, fps, score_median_shift)
        end = time.time()
        print('Time taken for median shift scoring is ', end - start)
        start = time.time()
        proposals = cas_to_proposals(cas, threshold_list, min_proposal_length, fps, score_all)
        end = time.time()
        print('Time taken for all scoring is ', end - start)
        self.assertTrue(True)
    
    # Also lets test the time difference between nms with and without merging
    def test_merge_timing(self):
        import time
        cas = torch.rand((20, 100, 9))
        threshold_list = [0.25, 0.5, 0.75]
        min_proposal_length = 5
        fps = 30
        seconds_to_index = fps / 16
        score_all = [[wide_short_scoring, stddev_scoring, median_shift_scoring], [1, 1, 1]]
        proposals = cas_to_proposals(cas, threshold_list, min_proposal_length, fps, score_all)
        nms_threshold = 0.5
        start = time.time()
        filtered_proposals = nms(proposals, nms_threshold, score_all, seconds_to_index, cas , False)
        end = time.time()
        print('Time taken for nms without merging is ', end - start)
        start = time.time()
        filtered_proposals = nms(proposals, nms_threshold, score_all, seconds_to_index, cas , True)
        end = time.time()
        print('Time taken for nms with merging is ', end - start)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()