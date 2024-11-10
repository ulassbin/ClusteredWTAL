import numpy as np
import torch
import copy



def interpolate_cas(cas, scale_factor):
    return cas # pass for now

def wide_short_scoring(batch_proposals, alpha=0.1):
    scored_proposals = [[] for _ in range(len(batch_proposals))]
    for i, prop in enumerate(batch_proposals):
        for j, item in enumerate(prop):
            if(len(item) != 4):
                continue
            class_id, start, end, data = item
            base_data = data[0]
            wide_data = data[-1]
            base_score = np.mean(base_data)
            edge_score = (np.sum(wide_data) - np.sum(base_data)) / (len(wide_data) - len(base_data) + 1e-6)
            score = base_score - alpha * edge_score # This will score the proposal, minus the edges. Making sharp edges less likely to be selected.
            scored_proposals[i].append([class_id, start, end, score])
    return scored_proposals

def stddev_scoring(batch_proposals, alpha=1): # Favors consistent proposals over fragmented ones
    scored_proposals = [[] for _ in range(len(batch_proposals))]
    for i, prop in enumerate(batch_proposals):
        for j, item in enumerate(prop):
            if(len(item) != 4):
                continue
            class_id, start, end, data = item
            base_data = data[0]
            base_score = np.mean(base_data)
            base_stddev = np.std(base_data)
            score = base_score / (base_stddev + alpha) # Smoother penalties when using 1 rather than using epsilon
            scored_proposals[i].append([class_id, start, end, score])
    return scored_proposals

def median_shift_scoring(batch_proposals): # Favors proposals that are centered
    scored_proposals = [[] for _ in range(len(batch_proposals))]
    for i, prop in enumerate(batch_proposals):
        for j, item in enumerate(prop):
            if(len(item) != 4):
                continue
            class_id, start, end, data = item
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
            scored_proposals[i].append([class_id, start, end, score])
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
            #                  class_id,               start,                  end,                    sum
            final_scores[i].append([score_list[0][i][j][0], score_list[0][i][j][1], score_list[0][i][j][2], score])
    return final_scores

def cas_to_proposals(cas, threshold_list, min_proposal_length, fps, score_config):
    batch = cas.shape[0]
    num_classes = cas.shape[2]
    index_to_seconds = 16 / fps
    borders = 0.1
    score_metrics, score_weights = score_config
    proposals = [[[] for _ in range(num_classes)] for _ in range(batch)]
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
                            if end - start > min_proposal_length:
                                current_length = end - start
                                wide_start = max(0, start - int(borders * current_length))
                                wide_end = min(time_length-1, end + int(borders * current_length))
                                proposal_data = [cas[k,start:end+1,i].cpu().numpy(), cas[k,wide_start:wide_end+1,i].cpu().numpy()]
                                batch_proposal[i].append([i, start*index_to_seconds, end*index_to_seconds, copy.deepcopy(proposal_data)])
                                print("Adding Proposal for class {} start {} end {}".format(i, start*index_to_seconds, end*index_to_seconds))
                            # reset start position
                            start = -1
                            end = -1
                if start != -1: # Finalize the proposal if it continues until the end of class_cas
                    if time_length - 1 - start > min_proposal_length: # +-1 index error is possible her (Check later.)
                        current_length = time_length - 1 - start
                        wide_start = max(0, start - int(borders * current_length))
                        wide_end = min(time_length-1, end + int(borders * current_length))
                        proposal_data = [cas[k,start:end+1,i].cpu().numpy(), cas[k,wide_start:wide_end+1,i].cpu().numpy()]
                        batch_proposal[i].append([i, start*index_to_seconds, end*index_to_seconds, copy.deepcopy(proposal_data)])
                        print("Adding Proposal for class {} start {} end {}".format(i, start*index_to_seconds, end*index_to_seconds))
            # Now lets score the batch_proposals
            scored_proposals = [[] for _ in range(num_classes)]
            scores = []
            print('Batch proposals are ', np.sum([len(proposal) for proposal in batch_proposal]))
            for j in range(len(score_metrics)):
                item = score_metrics[j](batch_proposal)
                #print('Score item is ', item)
                scores.append(score_metrics[j](batch_proposal))
            #print('Scores are', scores)
            scored_proposals = combine_scorings(scores, score_weights)
            #print("Scored proposals len ", len(scored_proposals))
            proposals[k].append(scored_proposals) # Assign computed proposal for batch k!
    return proposals # Proposals is a list with batch, num_classes, proposals


def checkMerges(proposal1, proposal2, score_config, t_factor, cas_batch):
    # Check if the proposals can be merged, if so 
    # calculate the merged score, and if it is better than the individual scores
    # return the merged proposal, otherwise return the original proposal with the highest score
    # proposal is [class_id, start, end, score]
    class_id1, start1, end1, score1 = proposal1
    class_id2, start2, end2, score2 = proposal2
    # Check if the proposals are of the same class
    if class_id1 != class_id2:
        if score1 > score2:
                return proposal1
        return proposal2
    intersection = max(0, min(end1, end2) - max(start1, start2)) 
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_start_index = int(union_start / t_factor)
    union_end_index = int(union_end / t_factor)
    borders = 0.1
    len_union = union_end - union_start
    union_start_index_wide = max(0, union_start_index - int(borders * len_union))
    union_end_index_wide = min(cas_batch[class_id1].shape[0]-1, union_end_index + int(borders * len_union))
    # Calculate the union score
    score_metrics, score_weights = score_config
    union_score = 0
    data = [cas_batch[class_id1][union_start_index:union_end_index+1], cas_batch[class_id1][union_start_index_wide:union_end_index_wide+1]]
    union_item = [[class_id1, union_start, union_end, data]]
    score_list = []
    for i in range(score_metrics):
        score_list.append(score_metrics[i](union_item))
    final_scores = combine_scorings(score_list, score_weights)
    union_score = final_scores[-1][-1]
    if union_score > score1 and union_score > score2: # We got a good merge!
        return [class_id1, union_start, union_end, union_score]
    elif score1 > score2:
        return proposal1
    else:
        return proposal2

def nms(proposals, nms_threshold, score_config, t_factor, cas , merging = True):
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
                            batch_final_proposals[j] = checkMerges(current_proposal, batch_final_proposals[j], score_config, t_factor, cas[batch_id])
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
    for batch_id in range(num_batch):
        batch_proposal = proposals[batch_id]
        batch_actionness = actionness[batch_id]
        for class_id in range(len(batch_proposal)):
            for proposal in batch_proposal[class_id]:
                print("Prop ", proposal)
                start_frame = int(proposal[1] * seconds_to_index)
                end_frame = int(proposal[2] * seconds_to_index)
                print(" Len {}, start {}, end {}".format(batch_actionness.shape, start_frame, end_frame))
                actionness_values = batch_actionness[start_frame:end_frame]
                if np.mean(actionness_values.cpu().numpy()) > threshold:
                    filtered_proposals[batch_id][class_id].append(proposal)
    return filtered_proposals
