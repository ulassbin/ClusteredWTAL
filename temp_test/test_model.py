import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(project_root, 'model'))
from config import cfg
# Add the model directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(project_root, 'model'))
from model import *
import helper_functions as helper
import data_loader as loader
from data_loader import NpyFeature
import torch
import numpy as np
import matplotlib.pyplot as plt

from logger import logging, CustomLogger
testLogger = CustomLogger(name='test_model')

import torch
from torch.utils.data import DataLoader

def custom_collate(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    temp_anno = [item[2] for item in batch]
    proposal_label = [item[3] for item in batch]
    file_name = [item[4] for item in batch]
    unpadded_video_length = [item[5] for item in batch]

    # Convert data and label to tensors
    data = torch.stack(data)
    label = np.array(label)
    temp_anno = np.array(temp_anno)

    return data, label, temp_anno, proposal_label, file_name, unpadded_video_length



def train_one_step(net, loader_iter, optimizer, criterion):
    net.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        data, label, temp_anno, proposal_label, file_name, unpadded_video_length = next(loader_iter)
    except StopIteration:
        raise ValueError("Loader iterator exhausted. Make sure the iterator is reset correctly in the training loop.")
    data, label = data.to(device), torch.tensor(label).to(device)
    video_scores, actionness, cas, base_vid_scores, base_actionnes, base_cas = net(data)
    cost = criterion(video_scores, label)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    return cost




@torch.no_grad()
def test_all(net, cfg, test_loader, test_info, step):
    net.eval()
    # data, label, temp_anno, proposal_label, file_name, unpadded_video_length
    # for test results format should be file_name, final_proposals, final_average_iou, vid_correspondences
    test_results = {}
    all_correspondences = []
    print_count = 0
    for data, label, temp_anno, label_proposals, file_name, vid_num_seg in test_loader:
        print_count += 1
        data, label = data.cuda(), torch.tensor(label).cuda()
        vid_num_seg = vid_num_seg[0]
        video_scores, actionness, cas, base_vid_scores, base_actionnes, base_cas = net(data)
        # Now we need to convert cas to proposals use the function in helper
        # We can use the function in helper_functions.py to convert cas to proposals
        proposals = helper.cas_to_proposals(cas, cfg.CAS_THRESH, cfg.MIN_PROPOSAL_LENGTH_INDEXWISE, cfg.FEATS_FPS)
        # calculate proposal length by summing the lengths of all class proposals and print
        filtered_proposals = helper.actionness_filter_proposals(proposals, actionness, cfg)
        final_proposals = helper.nms(filtered_proposals, cfg.NMS_THRESH) # non-maximum suppression
        final_average_iou, vid_correspondences = helper.calculate_IoU(final_proposals, label_proposals)
        test_results[file_name[0]] = [final_proposals, final_average_iou, vid_correspondences]

        if(print_count % 50 == 0):
            testLogger.log('Test Progress {}%'.format(print_count/len(test_loader) * 100.0), logging.WARNING)
            testLogger.log("Proposals length is {}".format(sum([len(proposal) for proposal in proposals])), logging.WARNING)
            testLogger.log("Filtered proposals length is {}".format(sum([len(proposal) for proposal in filtered_proposals])), logging.WARNING)
            testLogger.log('NMS      proposals length is {}'.format(sum([len(proposal) for proposal in final_proposals])), logging.WARNING)
            testLogger.log("Final average iou is {}".format(final_average_iou), logging.ERROR)
        # correspondences look like correspondences = [[proposal],[label],iou]] is a list!
        # I want to insert these [proposal],[label],iou] into all_correspondences
        all_correspondences += vid_correspondences 

    mAP_list, average_mAP = helper.calculate_mAp_from_correspondences(all_correspondences, cfg.NUM_CLASSES, cfg.TIOU_THRESH)
    testLogger.log("Average mAP is {}".format(average_mAP), logging.ERROR)
        #final_proposals = [utils.nms(v, cfg.NMS_THRESH) for _,v in proposal_dict.items()]
        #final_res['results'][vid[0]] = utils.result2json(final_proposals, cfg.CLASS_DICT)
        # Convert the proposals to json format somehow and save them to the final_res['results'][vid[0]] dictionary
    #json_path = os.path.join(cfg.OUTPUT_PATH, 'result.json')
    #json.dump(final_res, open(json_path, 'w'))

    test_info["step"].append(step)
    #test_info["test_acc"].append(acc.avg)
    test_info["average_mAP"].append(average_mAP)
    test_info['mApAll'].append(mAP_list)
    return test_info



class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()

    def forward(self, video_scores, label):
        label = label / torch.sum(label, dim=1, keepdim=True)
        loss = self.bce_criterion(video_scores, label)
        return loss



num_videos=None
simple_loader = loader.simpleLoader(cfg.VID_PATH, cfg.CLUSTER_PATH)
videos, feature_dim, frame_lengths, max_len = simple_loader.load_videos()
assert(feature_dim == cfg.FEATS_DIM)
cluster_labels, cluster_centers, cluster_center_indexes = simple_loader.load_cluster_information()

print('Feature dim {}, videos {}'.format(feature_dim, videos.shape))
print('Cluster labels {}, cluster_centers {}, cluster_center_indexes {}'.format(cluster_labels.shape, cluster_centers.shape, cluster_center_indexes.shape))

mainModel = CrashingVids(cfg, max_len, torch.tensor(cluster_centers).to('cuda'))


with torch.no_grad():
    video_scores, actionness, cas, base_vid_scores, base_actionnes, base_cas = mainModel(torch.tensor(videos[0:5,:]).to('cuda'))
    print(cas.shape)
    print(base_cas.shape)




dataset = NpyFeature(data_path=cfg.DATA_PATH, mode='train',
                    modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                    num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                    class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='None')

dataset.__getitem__(1)

train_loader = torch.utils.data.DataLoader(dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True, num_workers=1, # cfg.NUM_WORKERS,
        worker_init_fn=None, collate_fn=custom_collate)


test_dataset = NpyFeature(data_path=cfg.DATA_PATH, mode='train', # change mode to test later!
                    modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                    num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                    class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='None')

test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=1,
        shuffle=False, num_workers=1, # cfg.NUM_WORKERS,
        worker_init_fn=None, collate_fn=custom_collate) # Ok that worked!


#loader_iter = iter(train_loader)


# Define a function to write items in test info to a file
# dont overwrite each time append!
def write_test_info(test_info, file_path):
    with open(file_path, 'a') as f:
        f.write("Step: {}\n".format(test_info["step"][-1]))
        #f.write("Test_acc: {:.4f}\n".format(test_info["test_acc"][-1]))
        f.write("average_mAP: {:.4f}\n".format(test_info["average_mAP"][-1]))
        tIoU_thresh = np.linspace(0.1, 0.7, 7)
        for i in range(len(tIoU_thresh)):
            f.write("mAP@{}: {}\n".format(tIoU_thresh[i], test_info["mApAll"][-1][i]))
        f.close()


#data, label, temp_annotations, something, vid_num_seg = next(loader_iter)
#print("Data shape {} label shape {}".format(data.shape, label.shape))
#print('Temp temp_annotations {}'.format(vid_num_seg))
cfg.LR = eval(cfg.LR)
optimizer = torch.optim.Adam(mainModel.parameters(), lr=cfg.LR[0],
    betas=(0.9, 0.999), weight_decay=0.0005)

criterion = ActionLoss()


# Define a function to plot loss functions wait for 5-10 seconds and then close the plot
def plot_loss(losses):
    plt.plot(losses)
    plt.show(block=False)
    plt.pause(5)
    plt.close()

# Training loop
# Keep a history of epochwise losses
epoch_losses = []
test_info = {"step": [], "average_mAP": [], 'mApAll': []}
for epoch in range(cfg.NUM_EPOCHS):
    testLogger.log("----Epoch {}----".format(epoch), logging.ERROR)
    loader_iter = iter(train_loader) # Reset iterator
    epoch_loss = 0
    batch_losses = []
    print('Passed here')
    for step, batch in enumerate(loader_iter, start=1):
        # Skipping lr adjustment.
        testLogger.log("Training {}%".format(step/len(train_loader) * 100.0))
        cost = train_one_step(mainModel, loader_iter, optimizer, criterion)
        batch_losses.append(cost.item())
        epoch_loss += cost.item()
        if step == 1 or step % cfg.PRINT_FREQ == 0:
            # write an inline if statement to calculate mean_epoch_loss, if length of epoch_losses is 0 return 0
            mean_epoch_loss = np.mean(epoch_losses) if len(epoch_losses) > 0 else 0
            testLogger.log(('Epoch: [{0:04d}/{1}]\t' \
                    'Batch Loss {loss:.4f} Epoch Loss({loss_avg:.4f})\t'.format(
                    epoch, len(train_loader), loss=np.mean(batch_losses), loss_avg=mean_epoch_loss)), logging.WARNING)
    epoch_losses.append(epoch_loss)
    if epoch % cfg.TEST_FREQ == (cfg.TEST_FREQ - 1):
        testLogger.log("Testing at step {}".format(epoch))
        test_info = test_all(mainModel, cfg, test_loader, test_info, epoch)
        # Write the test results to a file, also print Iou Map@0.5
        #print("Test info is {}".format(test_info))
        print("Test info average mAP is {}".format(test_info['average_mAP'][-1]))
        print("Test info mApAll is {}".format(test_info['mApAll'][-1]))
        write_test_info(test_info, os.path.join(cfg.OUTPUT_PATH, "best_results.txt"))
        plot_loss(epoch_losses)

