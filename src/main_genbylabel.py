import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

import random
import argparse
import warnings
import h5py
import yaml
from tqdm import tqdm
import math
import pdb
import logging
import numpy as np

import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# from data_preprocess import data_prepare_generate_uschad
from Models.diffusion import GaussianProcess

import sys
# sys.path.append(os.path.join(os.path.dirname('__file__'), '../'))
from Utils.context_fid import Context_FID
from Utils.metric_utils import display_scores
from Utils.discriminative_metrics_torch import discriminative_score_metrics
from Utils.predictive_metrics_torch import predictive_score_metrics

from Utils.cross_correlation import CrossCorrelLoss

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# pdb.set_trace()

"""
remaining question:
turns out I still couldn't generate very long datas like the original one, need to 
figure out how to accomplish that
"""

"""
version: 1.0, start_date: Aug 31th
simplest version, make all the modules functional
including: 
1. train, load, sample
2. evaluation and visulization of the generated samples

version: 1.1 
Change the backbone into the pure Unet
result won't fly to the up
so it's the diffusion process's problem

"""


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch Training Script')
    
    parser.add_argument('--seed', type=int, default=888)
    parser.add_argument('--gpu', type=int, default=6,
                        help='GPU id to use, If given, only the specific gpu will be used, and'
                        'ddp will be disabled')
    parser.add_argument('--configname', type=str, default='gpto1',
                        help='path of config file')
    parser.add_argument('--train', action='store_true',
                        help='Set this flag to run training mode; omit for testing mode')
    parser.add_argument('--foldername', type=str,
                        help='experiment folder name')
    parser.add_argument('--testid', type=int, default=8,
                        help='Specify which pytorch parameter')
    parser.add_argument('--samplecnt', type=int, default=2,
                        help='how many samples of each activity')
    parser.add_argument('--resultfolder', default='../Experiments_genbylabel_seed2')
    parser.add_argument('--activityname', type=str, default='walkingforward',
                        help='activity name')
    
    args = parser.parse_args()
    return args


def seed_everything(seed, cudnn_deterministic=False):
    if seed is not None:
        logging.info(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config


"""
Data, the input data should be (n, win, 6)
get the prior and posterior directly from Dataset
"""


def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5

class GenerateDataset(Dataset):
    def __init__(self, args, config) -> None:
        super().__init__()
        self.data_path = config['dataloader']['dataset_path']
        # self.result_folder = config['dataloader']['result_folder']
        self.result_folder = args.resultfolder
        self.activityname = args.activityname

        self.dir = os.path.join(self.result_folder, args.foldername)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir, exist_ok=True)
        
        self.window = config['dataloader']['window_size']
        _, self.scaler = self.read_data()

        """
        samples and labels are already splitted and saved
        get them and use them directly
        """
        self.samples = self.dataset_genbylabel
        self.sample_num = self.samples.shape[0]
        
        self.labels = self.label_genbylabel
        
    def __get_samples(self, data):
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        for i in range(0, self.sample_num_total):
            # pdb.set_trace()
            start = i * self.window
            end = (i + 1) * self.window
            x[i, :, :] = data[start:end, :]
        
        # pdb.set_trace()
        np.save(os.path.join(self.dir, f"{self.activityname}_ground_truth_{self.window}_train.npy"), self.unnormalize(x))
        np.save(os.path.join(self.dir, f"{self.activityname}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(x))

        return x
        
    def read_data(self):
        with h5py.File(self.data_path, 'r') as f_r:
            data_grp = f_r['datas']
            dataset = data_grp[self.activityname][:]
            
            self.dataset_genbylabel = f_r['data_genbylabel'][:]
            self.label_genbylabel = f_r['label_genbylabel'][:]
        
        scaler = MinMaxScaler()
        scaler = scaler.fit(dataset)
        return dataset, scaler
    
    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)
    
    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        data = normalize_to_neg_one_to_one(data)
        return data
    
    def __unnormalize(self, data):
        x = unnormalize_to_zero_to_one(data)
        return self.scaler.inverse_transform(x)
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, index):
        x = self.samples[index, :, :]
        label = self.labels[index]
        return torch.tensor(label, dtype=torch.long), torch.from_numpy(x).float()


"""
Training method
"""
class Trainer:
    def __init__(self, config, args, device) -> None:
        self.config = config
        self.args = args
        # self.model = model
        self.device = device
        self._build()
    
    def train(self):
        self.model.train()
        min_eval_loss = float('inf')
        for epoch in range(1, self.config['solver']['num_epochs']+1):
            epoch_loss = 0.0
            for context, future in tqdm(self.train_loader,bar_format="{l_bar}{bar:20}{r_bar}",leave=True):
                context, future = context.to(self.device), future.to(self.device)
                self.optimizer.zero_grad()
                # pdb.set_trace()
                loss = self.model.get_loss_generate(context, future)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(self.train_loader)
            if min_eval_loss > epoch_loss:
                min_eval_loss = epoch_loss
                check_point = {
                    'ddpm': self.model.state_dict()
                }
                result_folder = self.args.resultfolder
                # pt_folder = os.path.join(self.config['dataloader']['result_folder'], self.args.foldername)
                pt_folder = os.path.join(result_folder, self.args.foldername)
                pt_path = os.path.join(pt_folder, f"genbylabel_ep{epoch}.pt")
                
            if epoch % self.config['solver']['eval_each'] == 0:
                torch.save(check_point, pt_path)
            
            logging.info(f"Epoch [{epoch}/{self.config['solver']['num_epochs']}], loss: {epoch_loss:.4f}")
    
    def _build_train_loader(self):
        trainset = GenerateDataset(self.args, self.config)
        self.train_loader = DataLoader(trainset, batch_size=self.config['dataloader']['train_batch_size'], shuffle=True)
        logging.info('> Train loader built!')
    
    def _build_model(self):
        self.model = GaussianProcess(self.config, self.device)
        
    def _build_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['solver']['learning_rate'], weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        logging.info('> Optimizer built!')
        
    def _build_criterion(self):
        self.criterion = nn.MSELoss()

    def _build(self):
        self._build_train_loader()
        self._build_criterion()
        self._build_model()
        self._build_optimizer()
        
        logging.info(r'> Everything built, have fun :) ')

"""
evaluation matrix class
input: 
predict is the generated sample
label is the true sample
"""


"""
Tester still need to change

"""

class Tester:
    # def __init__(self, config, args, device, checkpoint, result_path) -> None:
    def __init__(self, config, args, device) -> None:
        self.config = config
        self.args = args
        self.device = device
        self.activityname = args.activityname
        self.folder_path = os.path.join(args.resultfolder, args.foldername)
        pt_path, testid = self._find_recent_pth()
        
        self.result_path = os.path.join(self.folder_path, f"{args.activityname}_genbylabel.npy")
        self.checkpoint = os.path.join(self.folder_path, pt_path)
        
        logging.info(f"Using checkpoint: {self.checkpoint}")
        logging.info(f"Using result path: {self.result_path}")
        logging.info(f"Using activity: {self.activityname}")
        
        self.scaler_dict = dict()
        # self.result_folder = config['dataloader']['result_folder']
        self.result_folder = args.resultfolder
        self._build()
    
    def _find_recent_pth(self):
        # List all files in the directory
        # pdb.set_trace()
        files = os.listdir(self.folder_path)
        
        # Filter out .pt files
        pt_files = [f for f in files if 'genbylabel' in f and f.endswith('.pt')]
        
        # If no .pt files found, return None
        if not pt_files:
            return None

        # Extract the episode number from the filename using regular expressions
        def extract_episode_number(filename):
            match = re.search(r'ep(\d+)', filename)
            if match:
                return int(match.group(1))
            return -1
        
        # Sort the files by the extracted episode number
        pt_files.sort(key=extract_episode_number, reverse=True)

        testid = pt_files[0].split('.')[0].split('ep')[1]
        # Return the latest .pt file (the first one after sorting)
        return pt_files[0], testid
    
    def test(self):
        sample_list = []
        
        with torch.no_grad():
            for context in tqdm(self.activity_batches,bar_format="{l_bar}{bar:20}{r_bar}",leave=True):
                # context = torch.zeros_like(context).to(self.device)
                context = context.to(self.device)
                # pdb.set_trace()
                _, samples = self.model.sample_generate(context)
                sample_list.append(samples.detach().cpu().numpy())
                
        samples = np.vstack(sample_list)
        samples = unnormalize_to_zero_to_one(samples)
        
        # print(samples.shape)
        # result_path = os.path.join(self.folder_path, self.activityname+'.npy')
        np.save(self.result_path , samples)
        # self.result_path = result_path

    def random_choice(self, size, num_select=100):
        select_idx = np.random.randint(low=0, high=size, size=(num_select,))
        return select_idx

    def _create_sample_list(self):
        activity_label = {
            'walkingforward': 1,
            'walkingleft': 2,
            'walkingright': 3,
            'walkingupstairs': 4,
            'walkingdownstairs': 5,
            'runningforward': 6,
            'jumping': 7,
            'sitting': 8,
            'standing': 9,
            'sleeping': 10,
            'elevatorup': 11,
            'elevatordown': 12
        }
        activity_label_num = activity_label[self.activityname]-1
        
        data_path = self.config['dataloader']['dataset_path']
        with h5py.File(data_path, 'r') as f_r:
            datagrp = f_r['datas']
            label_list = f_r['label_genbylabel'][:]
            activity_count = np.sum(label_list == activity_label[self.activityname]-1)
        self.activity_list = torch.full((activity_count,), activity_label_num, dtype=torch.long)
        # pdb.set_trace()
        batch_size = self.config['dataloader']['eval_batch_size']
        self.activity_batches = torch.split(self.activity_list, batch_size)

    def show_results(self, json_path):
        # pdb.set_trace()
        """
        Context-FID score
        """
        logging.info(f"=====================================================")
        logging.info(f"Below is the Metrics of {self.activityname}!")
        logging.info(f"=====================================================")
        window_length = self.config['dataloader']['window_size']
        iterations = 5
        
        """
        shouldn't be like this, but for now, just specify the path
        """
        ori_data_path = os.path.join('../Experiments_100', self.activityname, f'{self.args.activityname}_norm_truth_{window_length}_train.npy')
        
        # self.ori_data = np.load(os.path.join(self.folder_path, f'{self.args.activityname}_norm_truth_24_train.npy'))
        self.ori_data = np.load(ori_data_path)
        self.fake_data = np.load(self.result_path)
        
        result_dict = {}
        context_fid_score = []

        for i in range(iterations):
            context_fid = Context_FID(self.ori_data[:], self.fake_data[:self.ori_data.shape[0]])
            context_fid_score.append(context_fid)
            logging.info(f"Iter {i}: context-fid = {context_fid}")
            
        fid_mean, fid_sigma = display_scores(context_fid_score)
        logging.info(f"Final context-fid score: {fid_mean}, {fid_sigma}")
        
        fid_dict = {}
        fid_dict['mean'] = fid_mean
        fid_dict['sigma'] = fid_sigma
        result_dict['context-fid'] = fid_dict
        
        # contextfid_score_mean = display_scores(context_fid_score)
        # logging.info(f"Final context-fid score : , {contextfid_score_mean}")
        
        """
        Correlation score
        """
        x_real = torch.from_numpy(self.ori_data)
        x_fake = torch.from_numpy(self.fake_data)
        correlational_score = []
        size = int(x_real.shape[0] / iterations)

        for i in range(iterations):
            real_idx = self.random_choice(x_real.shape[0], size)
            fake_idx = self.random_choice(x_fake.shape[0], size)
            corr = CrossCorrelLoss(x_real[real_idx, :, :], name='CrossCorrelLoss')
            loss = corr.compute(x_fake[fake_idx, :, :])
            correlational_score.append(loss.item())
            logging.info(f"Iter {i}: , cross-correlation = , {loss.item()}")
            
        corr_mean, corr_sigma = display_scores(correlational_score)
        logging.info(f"Final context-fid score: {corr_mean}, {corr_sigma}")

        corr_dict = {}
        corr_dict['mean'] = corr_mean
        corr_dict['sigma'] = corr_sigma
        result_dict['correlation'] = corr_dict

        # corr_score_mean = display_scores(correlational_score)
        # logging.info(f"Final correlation score : , {corr_score_mean}")
    
        """
        Discriminative score and Predictive score
        """
        discriminative_score = []

        for i in range(iterations):
            # temp_disc, fake_acc, real_acc = discriminative_score_metrics(self.ori_data[:], self.fake_data[:self.ori_data.shape[0]])
            temp_disc = discriminative_score_metrics(self.ori_data[:], self.fake_data[:self.ori_data.shape[0]], self.device)
            discriminative_score.append(temp_disc)
            # logging.info(f'Iter {i}: ', temp_disc, ',', fake_acc, ',', real_acc, '\n')
            logging.info(f"Iter {i}: , {temp_disc}")
        
        disc_mean, disc_sigma = display_scores(discriminative_score)
        logging.info(f"Final discriminative score : {disc_mean}, {disc_sigma}")

        disc_dict = {}
        disc_dict['mean'] = disc_mean
        disc_dict['sigma'] = disc_sigma
        result_dict['discriminative'] = disc_dict

        # discriminative_score_mean = display_scores(discriminative_score)
        # logging.info(f"Final discriminative score : {discriminative_score_mean}")
        
        predictive_score = []
        for i in range(iterations):
            temp_pred = predictive_score_metrics(self.ori_data, self.fake_data[:self.ori_data.shape[0]], self.device)
            predictive_score.append(temp_pred)
            logging.info(f"Iter {i}, epoch : {temp_pred}")
        
        pred_mean, pred_sigma = display_scores(predictive_score)
        logging.info(f"Final discriminative score : {pred_mean}, {pred_sigma}")
        
        pred_dict = {}
        pred_dict['mean'] = pred_mean
        pred_dict['sigma'] = pred_sigma
        result_dict['predictive'] = pred_dict
        
        # pred_score_mean = display_scores(predictive_score)
        # logging.info(f"Final Predictive score : {pred_score_mean}")

        with open(json_path, 'r') as file:
            file_dict = json.load(file)
        
        file_dict[self.activityname] = result_dict
        
        with open(json_path, 'w') as file:
            json.dump(file_dict, file, indent=4)
    
    def visualization(self, analysis='tsne'):
        """Using PCA or tSNE for generated and original data visualization.
        
        Args:
            - ori_data: original data
            - generated_data: generated synthetic data
            - analysis: tsne or pca
        """  
        window_length = self.config['dataloader']['window_size']
        # ori_data_path = os.path.join(self.args.resultfolder, self.activityname, f'{self.args.activityname}_norm_truth_{window_length}_train.npy')
        ori_data_path = os.path.join('../Experiments_100', self.activityname, f'{self.args.activityname}_norm_truth_{window_length}_train.npy')
        
        # self.ori_data = np.load(os.path.join(self.folder_path, f'{self.args.activityname}_norm_truth_24_train.npy'))
        self.ori_data = np.load(ori_data_path)
        self.fake_data = np.load(self.result_path)
        # pdb.set_trace()
        
        ori_data = self.ori_data
        generated_data = self.fake_data
        
        # Analysis sample size (for faster computation)
        anal_sample_no = min([1000, len(ori_data)])
        idx = np.random.permutation(len(ori_data))[:anal_sample_no]
            
        # Data preprocessing
        ori_data = np.asarray(ori_data)
        generated_data = np.asarray(generated_data)  
        
        ori_data = ori_data[idx]
        generated_data = generated_data[idx]
        
        no, seq_len, dim = ori_data.shape  
        
        for i in range(anal_sample_no):
            if (i == 0):
                prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
                prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
            else:
                prep_data = np.concatenate((prep_data, 
                                            np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
                prep_data_hat = np.concatenate((prep_data_hat, 
                                                np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
            
        # Visualization parameter        
        colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    
            
        if analysis == 'pca':
            # PCA Analysis
            pca = PCA(n_components = 2)
            pca.fit(prep_data)
            pca_results = pca.transform(prep_data)
            pca_hat_results = pca.transform(prep_data_hat)
            
            # Plotting
            f, ax = plt.subplots(1)    
            plt.scatter(pca_results[:,0], pca_results[:,1],
                        c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
            plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                        c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
        
            ax.legend()  
            plt.title('PCA plot')
            plt.xlabel('x-pca')
            plt.ylabel('y_pca')
            plt.show()
            
        elif analysis == 'tsne':
            
            # Do t-SNE Analysis together       
            prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
            
            # TSNE anlaysis
            tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
            tsne_results = tsne.fit_transform(prep_data_final)
            
            # Plotting
            f, ax = plt.subplots(1)
            
            plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                        c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
            plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                        c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
        
            ax.legend()
            
            img_path = os.path.join(self.folder_path, self.activityname+'_genbylabel.png')
            
            plt.title('t-SNE plot')
            plt.xlabel('x-tsne')
            plt.ylabel('y_tsne')
            plt.savefig(img_path, format='png', dpi=300)
            plt.show()
            

    def _build(self):
        self._build_model()
        self._create_sample_list()
        # self._get_samples()
    
    def _get_samples(self):
        dataset = GenerateDataset(self.args, self.config)
        self.testloader = DataLoader(dataset, batch_size=2048, shuffle=False)
        
    def _build_model(self):
        self.model = GaussianProcess(self.config, self.device)
        checkpoint = torch.load(self.checkpoint)
        self.model.load_state_dict(checkpoint['ddpm'])



"""
show results, such as distribution points
"""
def evaluation():
    pass


def build_log(args):
    log_dir = os.path.join(args.resultfolder, args.foldername)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'log.txt')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info('> Logger built')


def main():
    
    """
    step1: analysis the argument
    """
    args = parse_args()
    build_log(args)
    
    if args.seed is not None:
        seed_everything(args.seed)
    device = torch.device('cpu')
    if args.gpu is not None and args.gpu >= 0 and args.gpu <= 7:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        logging.info(f"Using GPU: {args.gpu}")
    else:
        logging.info(f"GPU usage is disabled or invalid GPU id:{args.gpu}")
    
    # folder_path = './Exp_activities/' + args.foldername
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    
    """
    step3: analysis the config
    """
    config_path = '../Config/' + args.configname + '.yaml'
    config = load_yaml_config(config_path)
    
    """
    step4: prepare the data
    """
    # data_prepare_generate_uschad(config)
    
    # data_prepare_etth1(config)
    # data_prepare_static(config)
    # pdb.set_trace()
    
    
    if args.train:
        # pdb.set_trace()
        # logging.info('check args', args.train)
        trainer = Trainer(config, args, device)
        trainer.train()
    else:
        # folder_path = './Exp_activities/' + args.foldername
        # folder_path = 
        # checkpoint = folder_path + f"/{args.activityname}_ep{testid}.pt"
        # result_path = folder_path + f"/{args.activityname}_ep{testid}_result.h5"
        # tester = Tester(config, args, device, checkpoint, result_path)
        
        json_path = os.path.join(args.resultfolder, 'all_activities.json')
        if not os.path.exists(json_path):
            with open(json_path, 'w') as file:
                empty_dict = {}
                json.dump(empty_dict, file, indent=4)
        
        tester = Tester(config, args, device)
        tester.test()
        tester.show_results(json_path)
        tester.visualization(analysis='tsne')
    
    # result_show = evaluation()
    
    return

if __name__ == '__main__':
    main()
    
    
    
        
    
"""
ideas:

1. figure out a way to re scale the scaled data back to the original data
no matter the input or the output

possible solution: record both origin and scaled, when it comes to scale the prediction back to origin, do the scale.fit() again to get the max and min

2. need to find out why the true and predict come from one position(training), but the predict_future and true_future are that different

3. the window length could be bigger so that the sample could actually show the difference, perhaps 512?

"""