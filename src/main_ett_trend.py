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
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Model.models import SimpleCNN
from data_preprocess import data_prepare, data_prepare_static, data_prepare_etth1, data_prepare_etth1_168, data_prepare_etth1_norm_sample
from Model.diffusion import DiffSensor, DiffusionProcess, GaussianProcess

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
    
    parser.add_argument('--seed', type=int, default=42)
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
class CustomDataset(Dataset):
    def __init__(self, data_matrix, config):
        super().__init__()
        self.config = config
        self.data_matrix = data_matrix
        self.prior_size = config['dataloader']['prior_size']
        self.overlap = config['dataloader']['overlap']
        self.future_size = config['dataloader']['future_size']
    
    def __len__(self):
        return self.data_matrix.shape[0]
    
    """
    split the dataset into prior and future
    in order to specify the specific label and sample
    to better 
    """
    def __getitem__(self, idx):
        data_matrix_idx = self.data_matrix[idx,:]
        prior = data_matrix_idx[:self.prior_size, :]
        future = data_matrix_idx[-(self.future_size+self.overlap):, :]
        # future = data_matrix_idx[self.config['dataloader']['future_size']:, :]
        return torch.from_numpy(prior).float(), torch.from_numpy(future).float()
        # return torch.from_numpy(prior).double(), torch.from_numpy(future).double()


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
    
    def normalize_tensor(self, tensor1, tensor2):
        overlap = self.config['dataloader']['overlap']
        # future_size = self.config['dataloader']['future_size']
        seq_len = tensor1.shape[1]
        mean_ = torch.mean(tensor1[:, -overlap:, :], dim=1).unsqueeze(1)
        # std_ = torch.ones_like(torch.std(tensor1, dim=1).unsqueeze(1))
        std_ = torch.std(tensor1, dim=1).unsqueeze(1)
        tensor1_ret = (tensor1-mean_.repeat(1, seq_len, 1)) / (std_.repeat(1, seq_len, 1)+1e-5)
        
        seq_len = tensor2.shape[1]
        tensor2_ret = (tensor2-mean_.repeat(1, seq_len, 1)) / (std_.repeat(1, seq_len, 1)+1e-5)
        # return mean_, std_, tensor1_ret, tensor2_ret[:,-future_size:,:]
        return mean_, std_, tensor1_ret, tensor2_ret
    
    def train(self):
        self.model.train()
        mini_eval_loss = float('inf')
        for epoch in range(1, self.config['solver']['num_epochs']+1):
            epoch_loss = 0.0
            for context,future in tqdm(self.train_loader,bar_format="{l_bar}{bar:20}{r_bar}",leave=True):
                """
                context, future shape: (batch_size, seq_len, channel)
                """
                context, future = context.to(self.device), future.to(self.device)
                _, _, context_norm, future_norm = self.normalize_tensor(context, future)
                self.optimizer.zero_grad()
                loss = self.model.get_loss(context_norm, future_norm)
                # loss = self.model.get_loss_nodiff(context, future)
                # pdb.set_trace()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(self.train_loader)
            logging.info(f"Epoch [{epoch}/{self.config['solver']['num_epochs']}], loss_norm: {epoch_loss:.4f}")
            
            if epoch % self.config['solver']['eval_each'] == 0:
                eval_loss = self.evaluation(epoch)
                if mini_eval_loss > eval_loss:
                    mini_eval_loss = eval_loss
                    check_point = {
                        'ddpm': self.model.state_dict()
                    }
                    pt_path = './Experiments/'+self.args.foldername+'/'+f"{self.config['model']['model_name']}_ep{epoch}.pt"
                    torch.save(check_point, pt_path)

            # self.scheduler.step()
            
    @torch.no_grad()
    def evaluation(self, epoch):
        self.model.eval()
        eval_loss = 0.0
        eval_loss_ori = 0.0
        with torch.no_grad():
            for context, future in tqdm(self.eval_loader,bar_format="{l_bar}{bar:20}{r_bar}",leave=True):
                context, future = context.to(self.device), future.to(self.device)
                ctx_mean, ctx_std, context_norm, future_norm = self.normalize_tensor(context, future)
                _, future_pred_norm = self.model.sample(context_norm)
                
                loss = F.mse_loss(future_norm, future_pred_norm)
                eval_loss += loss.item()
                
                out_len = future_pred_norm.shape[1]
                future_pred_ori = future_pred_norm * ctx_std.repeat(1,out_len,1) + ctx_mean.repeat(1,out_len,1)
                loss = F.mse_loss(future[:,-self.config['dataloader']['future_size']:,:], future_pred_ori)
                eval_loss_ori += loss.item()
                
            eval_loss /= len(self.eval_loader)
            eval_loss_ori /= len(self.eval_loader)
        logging.info(f"Evaluation - Epoch [{epoch}/{self.config['solver']['num_epochs']}], sample_loss_norm: {eval_loss:.4f}, sample_loss_ori: {eval_loss_ori}")
        return eval_loss
    
    
    def train_new(self):
        self.model.train()
        min_train_loss = float('inf')
        min_eval_loss = float('inf')
        min_epoch = 0
        check_point = {}
        for epoch in range(1, self.config['solver']['num_epochs']+1):
            epoch_loss = 0.0
            for context,future in tqdm(self.train_loader,bar_format="{l_bar}{bar:20}{r_bar}",leave=True):
                context, future = context.to(self.device), future.to(self.device)
                _, _, context_norm, future_norm = self.normalize_tensor(context, future)
                
                self.optimizer.zero_grad()
                # loss = self.model.get_loss(context, future)
                loss = self.model.get_loss_wtrend(context_norm, future_norm)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(self.train_loader)
            logging.info(f"Epoch [{epoch}/{self.config['solver']['num_epochs']}], loss: {epoch_loss:.4f}")
            
            if min_train_loss > epoch_loss:
                min_train_loss = epoch_loss
                check_point = {
                    'ddpm': self.model.state_dict()
                }
                min_epoch = epoch
            
            if epoch % self.config['solver']['eval_each'] == 0:
                model_eval = GaussianProcess(self.config, self.device)
                model_eval.load_state_dict(check_point['ddpm'])
                eval_loss = 0.0
                eval_loss_ori = 0.0
                
                with torch.no_grad():
                    for context, future in tqdm(self.eval_loader,bar_format="{l_bar}{bar:20}{r_bar}",leave=True):
                        context, future = context.to(self.device), future.to(self.device)
                        ctx_mean, ctx_std, context_norm, future_norm = self.normalize_tensor(context, future)
                        _, future_pred_norm = self.model.sample(context_norm)
                        # _, future_pred = self.model.sample(context)
                        
                        loss = F.mse_loss(future_norm, future_pred_norm)
                        # loss = F.mse_loss(future, future_pred)
                        eval_loss += loss.item()
                        
                        
                        out_len = future_pred_norm.shape[1]
                        future_pred_ori = future_pred_norm * ctx_std.repeat(1,out_len,1) + ctx_mean.repeat(1,out_len,1)
                        # loss = F.mse_loss(future[:,-self.config['dataloader']['future_size']:,:], future_pred_ori)
                        loss_n = F.mse_loss(future, future_pred_ori)
                        eval_loss_ori += loss_n.item()
                        
                    eval_loss /= len(self.eval_loader)
                    eval_loss_ori /= len(self.eval_loader)
                    
                if min_eval_loss > eval_loss:
                    min_eval_loss = eval_loss
                    pt_path = './Experiments/'+self.args.foldername+'/'+f"{self.config['model']['model_name']}_ep{min_epoch}.pt"
                    torch.save(check_point, pt_path)
                    
                logging.info(f"Evaluation - Epoch [{epoch}/{self.config['solver']['num_epochs']}], train_min_epoch:{min_epoch}, sample_loss: {eval_loss:.4f}, sample_loss_ori: {eval_loss_ori:.4f}")
                    
    
    def _build_train_loader(self):
        with h5py.File(self.config['dataloader']['matrix_path'], 'r') as f_r:
            data_group = f_r['datas_train_eval']
            trainset = data_group['trainset'][:]
        trainset = CustomDataset(trainset, self.config)
        self.train_loader = DataLoader(trainset, batch_size=self.config['dataloader']['train_batch_size'], shuffle=True)
        logging.info('> Train loader built!')
    
    def _build_eval_loader(self):
        with h5py.File(self.config['dataloader']['matrix_path'], 'r') as f_r:
            data_group = f_r['datas_train_eval']
            evalset = data_group['evalset'][:]
        evalset = CustomDataset(evalset, self.config)
        self.eval_loader = DataLoader(evalset, batch_size=self.config['dataloader']['eval_batch_size'])
        logging.info('> Eval loader built!')
    
    def _build_model(self):
        # self.model = DiffSensor(self.config, self.args, self.device, self.criterion)
        # self.model = DiffusionProcess(self.config, self.args, self.device, self.criterion)
        self.model = GaussianProcess(self.config, self.device)
        
    def _build_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['solver']['learning_rate'], weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        logging.info('> Optimizer built!')
        
    def _build_criterion(self):
        self.criterion = nn.MSELoss()

    def _build(self):
        self._build_train_loader()
        self._build_eval_loader()
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
    def __init__(self, config, args, device, checkpoint, result_path) -> None:
        self.config = config
        self.args = args
        self.device = device
        self.checkpoint = checkpoint
        self.result_path = result_path
        self.scaler_dict = dict()
        self._build()
        
    def normalize_tensor(self, tensor1, tensor2):
        overlap = self.config['dataloader']['overlap']
        # future_size = self.config['dataloader']['future_size']
        seq_len = tensor1.shape[1]
        mean_ = torch.mean(tensor1[:, -overlap:, :], dim=1).unsqueeze(1)
        # std_ = torch.ones_like(torch.std(tensor1, dim=1).unsqueeze(1))
        std_ = torch.std(tensor1, dim=1).unsqueeze(1)
        tensor1_ret = (tensor1-mean_.repeat(1, seq_len, 1)) / (std_.repeat(1, seq_len, 1)+1e-5)
        
        seq_len = tensor2.shape[1]
        tensor2_ret = (tensor2-mean_.repeat(1, seq_len, 1)) / (std_.repeat(1, seq_len, 1)+1e-5)
        # return mean_, std_, tensor1_ret, tensor2_ret[:,-future_size:,:]
        return mean_, std_, tensor1_ret, tensor2_ret
        
    def test(self):
        test_sensor_data = dict()
        noise_loss_all, sample_loss_all = 0.0, 0.0
        sample_num = 0
        with torch.no_grad():
            for key in tqdm(self.test_group,bar_format="{l_bar}{bar:20}{r_bar}",leave=True):
                key_dict = self.test_group[key]
                # print('check key_dict', key_dict.keys())
                norm_slide = key_dict['norm_slide']
                # ori_slide = key_dict['ori_slide']
                
                # print('check ori slide', ori_slide.shape, norm_slide.shape)
                # print('number', norm_slide[10:20, 0], ori_slide[10:20, 0])
                
                prior_size = self.config['dataloader']['prior_size']
                future_size = self.config['dataloader']['future_size']
                overlap_size = self.config['dataloader']['overlap']
                
                context = torch.from_numpy(norm_slide[:prior_size, :]).to(torch.float32).to(self.device)
                future = torch.from_numpy(norm_slide[-(future_size+overlap_size):, :]).to(torch.float32).to(self.device)
                context, future = context.unsqueeze(0), future.unsqueeze(0)
                ctx_mean, ctx_std, context_norm, future_norm = self.normalize_tensor(context, future)
                
                _, future_pred_norm = self.model.sample(context_norm)
                # _, future_pred = self.model.sample(context)
                
                # loss = F.mse_loss(future_norm, future_pred_norm)
                # eval_loss += loss.item()
                
                out_len = future_pred_norm.shape[1]
                future_pred_ori = future_pred_norm * ctx_std.repeat(1,out_len,1) + ctx_mean.repeat(1,out_len,1)
                # loss = F.mse_loss(future[:,-self.config['dataloader']['future_size']:,:], future_pred_ori)
                loss_n = F.mse_loss(future, future_pred_ori)
                sample_loss_all += loss_n.item()
                
                
                
                # scaler = MinMaxScaler()
                # scaler.min_ = key_dict['scaler_min'][:]
                # scaler.scale_ = key_dict['scaler_scale'][:]
                # scaler.data_min_ = key_dict['scaler_data_min'][:]
                # scaler.data_max_ = key_dict['scaler_data_max'][:]
                # scaler.data_range_ = key_dict['scaler_data_range'][:]
                
                # scaler = StandardScaler()
                # scaler.mean_ = key_dict['scaler_mean'][:]
                # scaler.scale_ = key_dict['scaler_var'][:]
                

                # print('check', scaler.min_, scaler.scale_)
                # pdb.set_trace()
                
                # ori_shape = future_pred.shape
                
                future_pred_ori = future_pred_ori.cpu().detach().numpy()
                # future_pred = future_pred.cpu().detach().numpy()
                # future_pred_denorm = scaler.inverse_transform(future_pred.reshape(-1, ori_shape[-1])).reshape(ori_shape)
                
                future = future.cpu().detach().numpy()
                # future_denorm = scaler.inverse_transform(future.reshape(-1, ori_shape[-1])).reshape(ori_shape)
                
                # print(future_pred_denorm[0,10:20,0], future_denorm[0,10:20,0])
                # pdb.set_trace()
                
                # print('check shape', future_denorm.shape, future_pred_denorm.shape)
                
                # denorm_sample_loss_all += np.mean((future_pred_denorm[:,:,0] - future_denorm[:,:,0])**2)
                # denorm_sample_loss_all += F.mse_loss(torch.from_numpy(future_denorm), torch.from_numpy(future_pred_denorm))
                # print(future_pred_denorm[0,10:20,0], future_denorm[0,10:20,0])
                # pdb.set_trace()
                
                test_sensor_data[key] = dict()
                test_sensor_data[key]['norm_slide'] = key_dict['norm_slide']
                test_sensor_data[key]['future_pred'] = future_pred_ori
                
                test_sensor_data[key]['ori_slide'] = key_dict['ori_slide']
                # test_sensor_data[key]['future_pred'] = future_pred_denorm
                sample_num += 1
        
        # logging.info(f"Testing - sample_loss: {sample_loss_all / sample_num}, denormed_sample_loss: {denorm_sample_loss_all / sample_num}")
        logging.info(f"Testing - sample_loss: {sample_loss_all / sample_num}")
        
        
        with h5py.File(self.result_path, 'w') as f_w:
            result_group_all = f_w.create_group('test_results')
            for key in test_sensor_data:
                data_dict = test_sensor_data[key]
                key_group = result_group_all.create_group(key)
                for key_ in data_dict:
                    dataset = data_dict[key_]
                    key_group.create_dataset(name=key_, data=dataset)

    def _build(self):
        self._build_test_loader()
        self._build_criterion()
        self._build_model()
        
    def _build_test_loader(self):
        test_group = {}
        with h5py.File(self.config['dataloader']['matrix_path'], 'r') as f_r:
            data_group = f_r['testset_grp']
            for key in data_group:
                test_group[key] = dict()
                key_dict = data_group[key]
                test_group[key]['norm_slide'] = key_dict['norm_slide'][:]
                test_group[key]['ori_slide'] = key_dict['ori_slide'][:]
                
                # test_group[key]['scaler_min'] = key_dict['scaler']['min_'][:]
                # test_group[key]['scaler_scale'] = key_dict['scaler']['scale_'][:]
                # test_group[key]['scaler_data_min'] = key_dict['scaler']['data_min_'][:]
                # test_group[key]['scaler_data_max'] = key_dict['scaler']['data_max_'][:]
                # test_group[key]['scaler_data_range'] = key_dict['scaler']['data_range_'][:]
                
                # test_group[key]['scaler_mean'] = key_dict['scaler']['mean_'][:]
                # test_group[key]['scaler_var'] = key_dict['scaler']['std_'][:]

        self.test_group = test_group
    
    def _build_criterion(self):
        self.criterion = nn.MSELoss()
    
    def _build_model(self):
        # self.model = DiffusionProcess(self.config, self.args, self.device, self.criterion)
        self.model = GaussianProcess(self.config, self.device)
        # self.model = DiffSensor(self.config, self.args, self.device, self.criterion)
        # checkpoint = torch.load(self.checkpoint, map_location='cpu')
        checkpoint = torch.load(self.checkpoint)
        self.model.load_state_dict(checkpoint['ddpm'])



"""
show results, such as distribution points
"""
def evaluation():
    pass


def build_log(args):
    log_dir = './Experiments/'+args.foldername
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
    
    folder_path = './Experiments/' + args.foldername
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    """
    step3: analysis the config
    """
    config_path = './Config/' + args.configname + '.yaml'
    config = load_yaml_config(config_path)
    
    """
    step4: prepare the data
    """
    # data_prepare(config)
    # data_prepare_static(config)
    # pdb.set_trace()
    # data_prepare_etth1(config)
    data_prepare_etth1_norm_sample(config)
    
    
    if args.train:
        # pdb.set_trace()
        # logging.info('check args', args.train)
        trainer = Trainer(config, args, device)
        trainer.train_new()
    else:
        testid = args.testid
        folder_path = './Experiments/' + args.foldername
        checkpoint = folder_path + f"/{config['model']['model_name']}_ep{testid}.pt"
        result_path = folder_path + f"/{config['model']['model_name']}_ep{testid}_result.h5"
        tester = Tester(config, args, device, checkpoint, result_path)
        tester.test()
    
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