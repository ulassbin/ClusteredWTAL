import os
import json
import numpy as np
import torch
import random
import torch.utils.data as data
from logger import logging, CustomLogger

logger = CustomLogger('DataLoader')


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False



class NpyFeature(data.Dataset):
    def __init__(self, data_path, mode, modal, feature_fps, num_segments, sampling, class_dict, seed=-1, supervision='weak'):
        if seed >= 0:
            set_seed(seed)

        self.mode = mode
        self.modal = modal
        self.feature_fps = feature_fps
        self.num_segments = num_segments
        self.feature_dim =[]
        self.padded_data = []
        self.padded_names = []

        if self.modal == 'all':
            self.feature_path = []
            for _modal in ['rgb', 'flow']:
                self.feature_path.append(os.path.join(data_path, 'features', self.mode, _modal))
        else:
            self.feature_path = os.path.join(data_path, 'features', self.mode, self.modal)

        self.max_len = self.get_max_len(self.feature_path)
        split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.strip())
        split_file.close()
        logger.log('=> {} set has {} videos'.format(mode, len(self.vid_list)), logging.WARNING)

        anno_path = os.path.join(data_path, 'gt.json')
        anno_file = open(anno_path, 'r')
        self.anno = json.load(anno_file)
        anno_file.close()

        self.class_name_to_idx = class_dict
        self.num_classes = len(self.class_name_to_idx.keys())

        self.supervision = supervision
        self.sampling = sampling

    def get_max_len(self, data_dir, num_videos=None):
        logger.log("Feature path is {}".format(data_dir))
        video_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        feature_dim = np.load(video_files[0]).shape[1]
        
        # Limit the number of videos if specified
        if num_videos is not None and num_videos < len(video_files):
            video_files = random.sample(video_files, num_videos)
        
        videos = [np.load(file) for file in video_files]  # Load video data
        lengths = [video.shape[0] for video in videos]
        max_len = max(lengths)
        logger.log("Max len is {}".format(max_len), logging.WARNING)
        return max_len

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, vid_num_seg, sample_idx = self.get_data(index)
        label, temp_anno = self.get_label(index, vid_num_seg, sample_idx)

        return data, label, temp_anno, self.vid_list[index], vid_num_seg

    def get_data_preloaded(self, index):
        vid_name = self.vid_list[index]

    def get_label_preloaded(self, index):
        pass

    def get_data(self, index):
        vid_name = self.vid_list[index]

        vid_num_seg = 0

        if self.modal == 'all':
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                    vid_name + '.npy')).astype(np.float32)
            flow_feature = np.load(os.path.join(self.feature_path[1],
                                    vid_name + '.npy')).astype(np.float32)

            vid_num_seg = rgb_feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(rgb_feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(rgb_feature.shape[0])
            else:
                sample_idx = no_noise(rgb_feature.shape[0])

            rgb_feature = rgb_feature[sample_idx]
            flow_feature = flow_feature[sample_idx]
            feature = np.concatenate((rgb_feature, flow_feature), axis=1)
        else:
            feature = np.load(os.path.join(self.feature_path,
                                    vid_name + '.npy')).astype(np.float32)
            
            vid_num_seg = feature.shape[0]

            if self.sampling == 'random': # Temporal noise addition...
                sample_idx = self.random_perturb(feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(feature.shape[0])
            else:
                sample_idx = self.no_noise(feature.shape[0])


            feature = feature[sample_idx]
        
        feature = np.pad(feature, ((0, self.max_len - len(feature)), (0, 0)), mode='constant')
        feature_torch = torch.from_numpy(feature)
        #logger.log("Feature final shape {}".format(feature_torch.shape), logging.WARNING)
        return feature_torch, vid_num_seg, sample_idx

    def get_label(self, index, vid_num_seg, sample_idx):
        vid_name = self.vid_list[index]
        anno_list = self.anno['database'][vid_name]['annotations']
        label = np.zeros([self.num_classes], dtype=np.float32)

        classwise_anno = [[]] * self.num_classes

        for _anno in anno_list:
            label[self.class_name_to_idx[_anno['label']]] = 1
            classwise_anno[self.class_name_to_idx[_anno['label']]].append(_anno)

        if self.supervision == 'weak':
            return label, torch.Tensor(0)
        else:
            temp_anno = np.zeros([vid_num_seg, self.num_classes])
            t_factor = self.feature_fps / 16

            for class_idx in range(self.num_classes):
                if label[class_idx] != 1:
                    continue

                for _anno in classwise_anno[class_idx]:
                    tmp_start_sec = float(_anno['segment'][0])
                    tmp_end_sec = float(_anno['segment'][1])

                    tmp_start = round(tmp_start_sec * t_factor)
                    tmp_end = round(tmp_end_sec * t_factor)

                    temp_anno[tmp_start:tmp_end+1, class_idx] = 1

            temp_anno = temp_anno[sample_idx, :]

            return label, torch.from_numpy(temp_anno)

    def no_noise(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        
        # Return evenly spaced samples without perturbation
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples).astype(int)  # Simply round down the calculated indexes
        return samples


    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)


    def uniform_sampling(self, length):
        if length <= self.num_segments:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)


class simpleLoader():
    def __init__(self, data_dir, cluster_path):
        self.data_dir = data_dir
        self.cluster_path = cluster_path

    def load_videos(self, num_videos=None):
        video_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.npy')]
        feature_dim = np.load(video_files[0]).shape[1]
        # Limit the number of videos if specified
        if num_videos is not None and num_videos < len(video_files):
            video_files = random.sample(video_files, num_videos)
        videos = [np.load(file) for file in video_files] # Flatten is necessary for DBSCAN
        lengths = [video.shape[0] for video in videos]
        max_len = max(lengths)
        padded_videos = np.array([np.pad(video, ((0, max_len - video.shape[0]), (0, 0)), 'constant').flatten() for video in videos])
        logger.log("Max len is {}".format(max_len), logging.WARNING)
        return padded_videos, feature_dim, lengths, max_len

    def load_cluster_information(self):
        labels = cluster_centers = cluster_center_indexes = None
        logger.log('{}/cluster_labels.npy'.format(self.cluster_path))
        if (os.path.exists('{}/cluster_labels.npy'.format(self.cluster_path))):
            labels = np.load('{}/cluster_labels.npy'.format(self.cluster_path))
        if (os.path.exists('{}/cluster_centers.npy'.format(self.cluster_path))):    
            cluster_centers = np.load('{}/cluster_centers.npy'.format(self.cluster_path))
        if(os.path.exists('{}/cluster_center_indexes.npy'.format(self.cluster_path))):    
            cluster_center_indexes = np.load('{}/cluster_center_indexes.npy'.format(self.cluster_path))
        return labels, cluster_centers, cluster_center_indexes