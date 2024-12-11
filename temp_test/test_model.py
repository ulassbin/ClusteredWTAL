import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Set project root and add necessary paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # cant I just au
sys.path.append(os.path.join(project_root, 'model'))
sys.path.append(os.path.join(project_root, 'cola_adapter'))

# Import custom modules
from config import cfg
from model import *
from loss import *
import helper_functions as helper
from helper_functions import OverlapWizard
from PropGenie import *
import data_loader as loader
from data_loader import NpyFeature
from logger import logging, CustomLogger

# import cola adapter and the metrics adapter
from cola_adapter import MetricsAdapter
from utils_cola import AverageMeter
# Initialize logger

# Memory management
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



testLogger = CustomLogger(name='test_model')

def custom_collate(batch):
  data = [item[0] for item in batch]
  label = [item[1] for item in batch]
  temp_anno = [item[2] for item in batch]
  proposal_label = [item[3] for item in batch]
  file_name = [item[4] for item in batch]
  unpadded_video_length = [item[5] for item in batch]
  data = torch.stack(data)
  label = np.array(label)
  temp_anno = np.array(temp_anno)
  return data, label, temp_anno, proposal_label, file_name, unpadded_video_length

def train_one_step(net, batch, optimizer, criterion_list, cfg):
  net.train()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  with torch.no_grad():
    batch_size = len(batch[0])
    cluster_embeddings = net.getClusterEmbeddings()
    cluster_embeddings_distilled = torch.mean(cluster_embeddings, dim=1)
    cluster_embeddings_expanded = cluster_embeddings_distilled.unsqueeze(0).repeat(batch_size, 1, 1)
  try:
    data, label, temp_anno, proposal_label, file_name, unpadded_video_length = batch
  except StopIteration:
    raise ValueError("Loader iterator exhausted. Make sure the iterator is reset correctly in the training loop.")
  data, label = data.to(device), torch.tensor(label).to(device)
  helper.print_memory_usage()
  video_scores, actionness, cas, base_vid_scores, base_actionnes, base_cas, embeddings = net(data)
  helper.print_memory_usage()

  cost = 0
  cost_list = []
  optimizer.zero_grad()
  for key in criterion_list.keys():
    if 'auto' in key:
      embeddings_distilled = torch.mean(embeddings, dim=1)
      loss, pseudo_labels = criterion_list[key](embeddings_distilled, cluster_embeddings_distilled)
      current_cost = cfg.SELF_LEARN_SCALE * loss
    else:
      current_cost = criterion_list[key](video_scores, label)
    cost_list.append(current_cost)
    cost += current_cost
  cost.backward()
  optimizer.step()
  return cost, cost_list

def visualize_cas(cas, actionness, max_plots=5, file_name=None):
  num_plots = min(max_plots, cas.shape[0])
  fig, axs = plt.subplots(num_plots, 2, figsize=(16, 10 * num_plots))
  for batch_id in range(num_plots):
    cas_batch = cas[batch_id].cpu().numpy()
    actionness_batch = actionness[batch_id].cpu().numpy()
    if cas.shape[0] > 1:
      for class_id in range(cas_batch.shape[1]):
        axs[batch_id, 0].plot(cas_batch[:, class_id])
      axs[batch_id, 0].set_title(f'CAS for Batch {batch_id} {file_name}')
      axs[batch_id, 0].set_xlabel('Temporal Length')
      axs[batch_id, 0].set_ylabel('Class Activation Scores')
      axs[batch_id, 1].plot(actionness_batch, color='r')
      axs[batch_id, 1].set_title(f'Actionness for Batch {batch_id}')
      axs[batch_id, 1].set_xlabel('Temporal Length')
      axs[batch_id, 1].set_ylabel('Actionness Score')
    elif cas.shape[0] == 1:
      for class_id in range(cas_batch.shape[1]):
        axs[0].plot(cas_batch[:, class_id])
      axs[0].set_title(f'CAS for Batch {batch_id} {file_name}')
      axs[0].set_xlabel('Temporal Length')
      axs[0].set_ylabel('Class Activation Scores')
      axs[1].plot(actionness_batch, color='r')
      axs[1].set_title(f'Actionness for Batch {batch_id}')
      axs[1].set_xlabel('Temporal Length')
      axs[1].set_ylabel('Actionness Score')
  plt.tight_layout()
  plt.show()
  plt.pause(5)
  plt.close()

  return

@torch.no_grad()
def test_all(net, cfg, test_loader, test_info, step):
  net.eval()
  all_correspondences = []
  print_count = 0
  genie = ProposalGenie(cfg, cfg.SCORE_CONFIG)
  ow = OverlapWizard(cfg.TIOU_THRESH, cfg.NUM_CLASSES)
  metrics = MetricsAdapter()
  final_res = {'method': '[CoLA] https://github.com/zhang-can/CoLA', 'results': {}}
  acc = AverageMeter()

  for data, label, temp_anno, label_proposals, file_name, vid_num_seg in test_loader:
    print_count += 1
    batch_size = len(data)
    data, label = data.cuda(), torch.tensor(label).cuda()
    vid_num_seg = vid_num_seg[0]
    video_scores, actionness, cas, base_vid_scores, base_actionness, base_cas, embeddings = net(data) # this works with batch size already
    
    if(cfg.COLA_UTILS):
      final_res, acc = metrics.form_proposals(final_res, acc, cfg, label, video_scores, cas, actionness, file_name, vid_num_seg)
    else:
      proposals = genie.cas_to_proposals(cas)
      filtered_proposals = genie.filter_proposals(proposals, actionness)
      nms_proposals = genie.nms(cas, filtered_proposals)
      average_iou, vid_correspondences = ow.calculate_IoU(nms_proposals, label_proposals)
      testLogger.log("Props: Raw {}, Filtered {}, NMS {}".format(getProposalItemCount(proposals), getProposalItemCount(filtered_proposals), getProposalItemCount(nms_proposals)), logging.WARNING)
      testLogger.log("Batch average iou is {}".format(np.mean(average_iou)), logging.ERROR)
    testLogger.log('Test Progress {}%'.format(print_count/len(test_loader) * 100.0), logging.WARNING)
    visualize_cas(cas, actionness, 5, file_name)
  if(cfg.COLA_UTILS):
    json_path = metrics.write_results_to_json(final_res, cfg)
    mAP, average_mAp = metrics.getmAp(cfg, json_path)
    for i in range(cfg.TIOU_THRESH.shape[0]):
        test_info["mAP@{:.1f}".format(cfg.TIOU_THRESH[i])].append(mAP[i])
  else:
    mAp_list, average_mAp = ow.calculate_mAp_from_correspondences(cfg.TIOU_THRESH)
    test_info['mApAll'].append(mAp_list)
    testLogger.log("Average mAP is {}".format(average_mAp), logging.ERROR)
  test_info["step"].append(step)
  test_info["average_mAP"].append(average_mAp)
  return test_info

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run experiment with a specified name.')
  parser.add_argument(
    'exp_name',
    type=str,
    nargs='?',
    default='default',
    help='Name of the experiment (default: "default_experiment")'
  )
  args = parser.parse_args()
  exp_name = args.exp_name
  print(f'Running experiment: {exp_name}')

  writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_PATH, 'runs', exp_name))
  num_videos = None
  simple_loader = loader.simpleLoader(cfg.VID_PATH, cfg.CLUSTER_PATH)
  #videofiles, feature_dim, frame_lengths, max_len = simple_loader.load_videos()
  #assert(feature_dim == cfg.FEATS_DIM)
  cluster_labels, cluster_centers, cluster_center_indexes = simple_loader.load_cluster_information()

  #print('Feature dim {}, videos {}'.format(feature_dim, len(videofiles)))
  print('Cluster labels {}, cluster_centers {}, cluster_center_indexes {}'.format(cluster_labels.shape, cluster_centers.shape, cluster_center_indexes.shape))

  mainModel = CrashingVids(cfg, torch.tensor(cluster_centers).to('cuda'))

  dataset = NpyFeature(data_path=cfg.DATA_PATH, mode='train',
    modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
    num_segments=cfg.NUM_SEGMENTS, supervision='weak',
    class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='None', len_override=cfg.TEMPORAL_LENGTH, quick_run=cfg.QUICK_RUN)

  dataset.__getitem__(1)

  train_loader = torch.utils.data.DataLoader(dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True, num_workers=0,
    worker_init_fn=None, collate_fn=custom_collate)

  test_dataset = NpyFeature(data_path=cfg.DATA_PATH, mode='train',
    modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
    num_segments=cfg.NUM_SEGMENTS, supervision='weak',
    class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='None', len_override=cfg.TEMPORAL_LENGTH, quick_run=cfg.QUICK_RUN)

  test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=1,
    shuffle=False, num_workers=0,
    worker_init_fn=None, collate_fn=custom_collate)

  def write_test_info(test_info, loss, file_path, writer):
    with open(file_path, 'a') as f:
      f.write("Step: {}\n".format(test_info["step"][-1]))
      print('test_info average mAP is {}'.format(test_info['average_mAP']))
      f.write("average_mAP: {:.4f}\n".format(test_info["average_mAP"][-1]))
      writer.add_scalar('mAP/average', test_info["average_mAP"][-1], test_info["step"][-1])
      tIoU_thresh = np.linspace(0.1, 0.7, 7)
      for i in range(len(tIoU_thresh)):
        mAp_val = test_info["mAP@{:.1f}".format(cfg.TIOU_THRESH[i])][-1] # -1 to get the last value
        f.write("mAP@{}: {}\n".format(tIoU_thresh[i], mAp_val))
        print('Items to be written {} {}'.format(mAp_val, test_info["step"][-1]))
        writer.add_scalar('mAP/{}'.format(tIoU_thresh[i]), mAp_val, test_info["step"][-1])
      f.write('Loss: {}'.format(loss))
      f.close()

  cfg.LR = eval(cfg.LR)
  optimizer = torch.optim.Adam(mainModel.parameters(), lr=cfg.LR[0],
    betas=(0.9, 0.999), weight_decay=0.0005)

  autoLoss = AutoLabelClusterCrossEntropyLoss()
  criterion = ActionLoss()
  criterion_list = {'action': criterion, 'auto': autoLoss}

  def plot_loss(losses):
    plt.plot(losses)
    plt.show(block=False)
    plt.pause(20)
    plt.close()

  epoch_losses = []
  train_total_steps = len(train_loader)
  test_total_steps = len(test_loader)
  test_info = {"step": [], "average_mAP": [], 'mApAll': []}
  for item in cfg.TIOU_THRESH:
    test_info["mAP@{:.1f}".format(item)] = []
  for epoch in range(cfg.NUM_EPOCHS):
    testLogger.log("----Epoch {}----".format(epoch), logging.ERROR)
    loader_iter = iter(train_loader)
    epoch_loss = 0
    batch_losses = []
    epoch_loss_list = []
    for step, batch in enumerate(loader_iter, start=1):
      testLogger.log("Training {}%".format(step/len(train_loader) * 100.0))
      cost, cost_list = train_one_step(mainModel, batch, optimizer, criterion_list, cfg)
      torch.cuda.empty_cache()
      batch_losses.append(cost.item())
      epoch_loss_list.append([c.item() for c in cost_list])
      if step == 1 or step % cfg.PRINT_FREQ == 0:
        mean_epoch_loss = np.mean(epoch_losses) if len(epoch_losses) > 0 else 0
        testLogger.log(('Epoch: [{0:04d}/{1}]\t' \
          'Batch Loss {loss:.4f} Epoch Loss({loss_avg:.4f})\t'.format(
          epoch, len(train_loader), loss=np.mean(batch_losses), loss_avg=mean_epoch_loss)), logging.WARNING)
        writer.add_scalar('Loss/train', np.mean(batch_losses), epoch * train_total_steps + step)
        writer.flush()
    epoch_loss = np.sum(batch_losses)
    epoch_losses.append(epoch_loss)
    writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
    writer.add_scalar('Loss/train_epoch_action', np.sum([loss_list[0] for loss_list in epoch_loss_list ]), epoch)
    writer.add_scalar('Loss/train_epoch_auto', np.sum([loss_list[1] for loss_list in epoch_loss_list ]), epoch)
    if epoch % cfg.TEST_FREQ == (cfg.TEST_FREQ - 1):
      testLogger.log("Testing at step {}".format(epoch))
      test_info = test_all(mainModel, cfg, test_loader, test_info, epoch)
      print("Test info average mAP is {}".format(test_info['average_mAP'][-1]))
      #print("Test info mApAll is {}".format(test_info['mApAll'][-1]))
      write_test_info(test_info, epoch_losses[-1], os.path.join(cfg.OUTPUT_PATH, "best_results1.txt"), writer)
      #plot_loss(epoch_losses)
      torch.save(mainModel.state_dict(), os.path.join(cfg.OUTPUT_PATH, 'model_weights{}.pth'.format(epoch)))
    writer.flush()
  writer.close()
