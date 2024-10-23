import h5py
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pdb
"""
read data.h5
slide data into windows(samples)
put them into one matrix
"""
def normalize(dataset, scalar):
    # learn the min and max values
    scalar.fit(dataset)
    dataset_trans = scalar.transform(dataset)
    return dataset_trans


def sliding_window(dataset, window_size, step_size):
    data_len = dataset.shape[0]
    num_windows = (data_len - window_size) // step_size + 1
    windows = []
    for i in range(num_windows):
        start_idx = i * step_size
        windows.append(dataset[start_idx : start_idx + window_size, :])
    return np.array(windows)


def data_prepare(config):
    if os.path.exists(config['dataloader']['matrix_path']):
        return
    else:
        scaler = MinMaxScaler(feature_range=(-1,1))
        train_, eval_ = [],[]
        test_dict = {}
        with h5py.File(config['dataloader']['dataset_path'], 'r') as f_r:
            data_group = f_r['datas']
            for key in data_group:
                dataset_ori = data_group[key] # (n, 6)
                # print('check dataset i', dataset_i[0:100, 1])
                # pdb.set_trace()
                # dataset_norm = normalize(dataset_ori, scaler)
                dataset_norm = scaler.fit_transform(dataset_ori)
                if dataset_ori.shape[0] < config['dataloader']['window_size']:
                    print('data length < window size')
                    continue
                dataset_norm_slide = sliding_window(dataset_norm, config)
                
                subject_i = key.split('_')[0].split('b')[1]
                if int(subject_i) in config['dataloader']['train_list']:
                    train_.append(dataset_norm_slide)
                elif int(subject_i) in config['dataloader']['eval_list']:
                    eval_.append(dataset_norm_slide)
                # else:
                if int(subject_i) in config['dataloader']['test_list']:
                    test_dict[key] = {
                        'norm_slide': dataset_norm_slide,
                        'ori_slide': sliding_window(dataset_ori, config),
                        'scaler': scaler
                    }
                    
                    # dataset_ori_slide = sliding_window(dataset_ori, config)
                    # test_dict[key] = dict()
                    # test_dict[key]['norm_slide'] = dataset_norm_slide
                    # test_dict[key]['']
                    # test_dict[key]
                    # test_.append(dataset_i_slide)
        
        train_ = np.vstack(train_)
        eval_ = np.vstack(eval_)
        # test_ = np.vstack(test_)
        
        with h5py.File(config['dataloader']['matrix_path'], 'w') as f_w:
            data_group = f_w.create_group('datas_train_eval')
            data_group.create_dataset('trainset', data=train_)
            data_group.create_dataset('evalset', data=eval_)
            
            test_group = f_w.create_group('datas_test')
            for key, data in test_dict.items():
                key_group = test_group.create_group(key)
                key_group.create_dataset(name='norm_slide', data=data['norm_slide'])
                key_group.create_dataset(name='ori_slide', data=data['ori_slide'])
                
                scaler_group = key_group.create_group('scaler')
                scaler_group.create_dataset('min_', data=data['scaler'].min_)
                scaler_group.create_dataset('scale_', data=data['scaler'].scale_)
                scaler_group.create_dataset('data_min_', data=data['scaler'].data_min_)
                scaler_group.create_dataset('data_max_', data=data['scaler'].data_max_)
                scaler_group.create_dataset('data_range_', data=data['scaler'].data_range_)
                
                # scaler_group.create_dataset('mean_', data=data['scaler'].var_)
                # scaler_group.create_dataset('var_', data=data['scaler'].var_)
    return


def load_activity_parameters(h5py_path):
    """Loads the mean and std for each activity from the HDF5 file."""
    activity_para_dict = {}

    with h5py.File(h5py_path, 'r') as f_r:
        for activity in f_r.keys():
            mean = np.array(f_r[f'{activity}/mean'])
            std = np.array(f_r[f'{activity}/std'])
            activity_para_dict[activity] = {'mean': mean, 'std': std}
    
    return activity_para_dict


def apply_normalization(data, activity, activity_para_dict):
    """Applies the mean and std normalization for the given activity."""
    mean = activity_para_dict[activity]['mean']
    std = activity_para_dict[activity]['std']
    
    # Normalize the data
    return (data - mean) / (std + 1e-8)  # Adding a small constant to avoid division by zero


def data_prepare_uschad(config):
    if os.path.exists(config['dataloader']['matrix_path']):
        return
    else:
        train_, eval_ = [],[]
        test_dict = dict()
        window_size = config['dataloader']['window_size']
        
        uschad_para_path = './Datas/uschad_activity_parameters.h5'
        activity_para_dict = load_activity_parameters(uschad_para_path)
        
        with h5py.File(config['dataloader']['dataset_path'], 'r') as f_r:
            data_group = f_r['datas']
            for key in data_group:
                dataset_ori = data_group[key] # (n, 6)
                
                subject_i = int(key.split('_')[0].split('b')[1])
                activity_i = key.split('_')[1].split('a')[1]
                
                normalized_data_i = apply_normalization(dataset_ori, activity_i, activity_para_dict)
                
                if int(subject_i) in config['dataloader']['train_list']:
                    dataset_norm_slide = sliding_window(normalized_data_i, window_size, config['dataloader']['step_size_train'])
                    train_.append(dataset_norm_slide)
                elif int(subject_i) in config['dataloader']['eval_list']:
                    dataset_norm_slide = sliding_window(normalized_data_i, window_size, config['dataloader']['step_size_eval'])
                    eval_.append(dataset_norm_slide)

                if int(subject_i) in config['dataloader']['test_list']:
                    dataset_slide = sliding_window(dataset_ori, window_size, config['dataloader']['step_size_test'])
                    dataset_norm_slide = sliding_window(normalized_data_i, window_size, config['dataloader']['step_size_test'])
                    test_dict[key] = {
                        'norm_slide': dataset_norm_slide,
                        'ori_slide': dataset_slide,
                    }
                    
        train_ = np.vstack(train_)
        eval_ = np.vstack(eval_)
        
        with h5py.File(config['dataloader']['matrix_path'], 'w') as f_w:
            data_group = f_w.create_group('datas_train_eval')
            data_group.create_dataset('trainset', data=train_)
            data_group.create_dataset('evalset', data=eval_)
            
            test_group = f_w.create_group('datas_test')
            for key, data in test_dict.items():
                key_group = test_group.create_group(key)
                key_group.create_dataset(name='norm_slide', data=data['norm_slide'])
                key_group.create_dataset(name='ori_slide', data=data['ori_slide'])
    return


def data_prepare_static(config):
    if os.path.exists(config['dataloader']['matrix_path']):
        return
    else:
        # scaler = MinMaxScaler(feature_range=(-1,1))
        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        # file_path_dataset = os.path.join(folder_path, 'USC_HAD_dataset.h5')
        train_, eval_ = [],[]
        test_dict = {}
        with h5py.File(config['dataloader']['dataset_path'], 'r') as f_r:
            data_group = f_r['datas']
            for key in data_group:
                activity_i = int(key.split('_')[1][1:])
                if activity_i not in [8, 9, 10, 11, 12]:
                    continue
                
                # pdb.set_trace()
                
                dataset_ori = data_group[key] # (n, 6)
                # print('check dataset i', dataset_i[0:100, 1])
                # pdb.set_trace()
                # dataset_norm = normalize(dataset_ori, scaler)
                dataset_norm = scaler.fit_transform(dataset_ori)
                # print('check dataset i', dataset_i[0:100, 1])
                # pdb.set_trace()
                dataset_norm_slide = sliding_window(dataset_norm, config)
                
                subject_i = key.split('_')[0].split('b')[1]
                if int(subject_i) in config['dataloader']['train_list']:
                    train_.append(dataset_norm_slide)
                elif int(subject_i) in config['dataloader']['eval_list']:
                    eval_.append(dataset_norm_slide)
                # else:
                if int(subject_i) in config['dataloader']['test_list']:
                    test_dict[key] = {
                        'norm_slide': dataset_norm_slide,
                        'ori_slide': sliding_window(dataset_ori, config),
                        'scaler': scaler
                    }
                    
                    # dataset_ori_slide = sliding_window(dataset_ori, config)
                    # test_dict[key] = dict()
                    # test_dict[key]['norm_slide'] = dataset_norm_slide
                    # test_dict[key]['']
                    # test_dict[key]
                    # test_.append(dataset_i_slide)
        
        train_ = np.vstack(train_)
        eval_ = np.vstack(eval_)
        # test_ = np.vstack(test_)
        
        with h5py.File(config['dataloader']['matrix_path'], 'w') as f_w:
            data_group = f_w.create_group('datas_train_eval')
            data_group.create_dataset('trainset', data=train_)
            data_group.create_dataset('evalset', data=eval_)
            
            test_group = f_w.create_group('datas_test')
            for key, data in test_dict.items():
                key_group = test_group.create_group(key)
                key_group.create_dataset(name='norm_slide', data=data['norm_slide'])
                key_group.create_dataset(name='ori_slide', data=data['ori_slide'])
                
                scaler_group = key_group.create_group('scaler')
                scaler_group.create_dataset('min_', data=data['scaler'].min_)
                scaler_group.create_dataset('scale_', data=data['scaler'].scale_)
                scaler_group.create_dataset('data_min_', data=data['scaler'].data_min_)
                scaler_group.create_dataset('data_max_', data=data['scaler'].data_max_)
                scaler_group.create_dataset('data_range_', data=data['scaler'].data_range_)
                
                # scaler_group.create_dataset('mean_', data=data['scaler'].var_)
                # scaler_group.create_dataset('var_', data=data['scaler'].var_)
    
    return


def data_prepare_etth1(config):
    if os.path.exists(config['dataloader']['matrix_path']):
        return
    else:
        train_, eval_ = [],[]
        test_dict = {}
        with h5py.File(config['dataloader']['dataset_path'], 'r') as f_r:
            data_group = f_r['datas']
            train_ = data_group['trainset'][:]
            eval_ = data_group['valset'][:]
            test_ = data_group['testset'][:]
            for i in range(test_.shape[0]):
                test_dict[str(i+1)] = {
                    'norm_slide': test_[i,:],
                    'ori_slide': test_[i,:]
                }

        with h5py.File(config['dataloader']['matrix_path'], 'w') as f_w:
            data_group = f_w.create_group('datas_train_eval')
            data_group.create_dataset('trainset', data=train_)
            data_group.create_dataset('evalset', data=eval_)
            
            test_group = f_w.create_group('datas_test')
            for key, data in test_dict.items():
                key_group = test_group.create_group(key)
                key_group.create_dataset(name='norm_slide', data=data['norm_slide'])
                key_group.create_dataset(name='ori_slide', data=data['ori_slide'])
    return


def data_prepare_etth1_168(config):
    if os.path.exists(config['dataloader']['matrix_path']):
        return
    else:
        train_, eval_ = [],[]
        test_dict = {}
        with h5py.File(config['dataloader']['dataset_path'], 'r') as f_r:
            data_group = f_r['datas']
            train_ = data_group['trainset'][:]
            eval_ = data_group['valset'][:]
            
            
            scaler_group = f_r['scaler']
            scaler = MinMaxScaler()
            scaler.min_ = scaler_group['min_'][:]
            scaler.scale_ = scaler_group['scale_'][:]
            scaler.data_min_ = scaler_group['data_min_'][:]
            scaler.data_max_ = scaler_group['data_max_'][:]
            scaler.data_range_ = scaler_group['data_range_'][:]
            
            
            # scaler_group = f_r['scaler']
            test_ = data_group['testset'][:]
            for i in range(test_.shape[0]):
                norm_test = test_[i,:]
                ori_test = scaler.inverse_transform(norm_test)
                
                test_dict[str(i+1)] = {
                    'norm_slide': norm_test,
                    'ori_slide': ori_test,
                    'scaler': scaler
                }

        with h5py.File(config['dataloader']['matrix_path'], 'w') as f_w:
            data_group = f_w.create_group('datas_train_eval')
            data_group.create_dataset('trainset', data=train_)
            data_group.create_dataset('evalset', data=eval_)
            
            test_group = f_w.create_group('datas_test')
            for key, data in test_dict.items():
                key_group = test_group.create_group(key)
                key_group.create_dataset(name='norm_slide', data=data['norm_slide'])
                key_group.create_dataset(name='ori_slide', data=data['ori_slide'])
                
                scaler_group = key_group.create_group('scaler')
                scaler_group.create_dataset('min_', data=data['scaler'].min_)
                scaler_group.create_dataset('scale_', data=data['scaler'].scale_)
                scaler_group.create_dataset('data_min_', data=data['scaler'].data_min_)
                scaler_group.create_dataset('data_max_', data=data['scaler'].data_max_)
                scaler_group.create_dataset('data_range_', data=data['scaler'].data_range_)
    return


def data_prepare_etth1_norm_sample(config):
    if os.path.exists(config['dataloader']['matrix_path']):
        return
    else:
        test_dict = {}
        scaler_all = StandardScaler()
        # scaler1 = MinMaxScaler(feature_range=(-1, 1))
        # scaler = StandardScaler()
        with h5py.File(config['dataloader']['dataset_path'], 'r') as f_r:
            data_group = f_r['datas']
            trainset = data_group['trainset'][:]
            evalset = data_group['valset'][:]
            testset = data_group['testset'][:]
            
            """
            Before the windowed normalization, do a StandardScaler first
            """
            scaler_all.fit(trainset)
            trainset = scaler_all.transform(trainset)
            evalset = scaler_all.transform(evalset)
            testset = scaler_all.transform(testset)
            
            # pdb.set_trace()
            
            window_size = config['dataloader']['window_size']
            step_size_train = config['dataloader']['step_size_train']
            trainset_slide = sliding_window(trainset, window_size, step_size_train)
            trainset_slide_norm_array = np.array(trainset_slide)
            
            # trainset_slide_norm = [scaler.fit_transform(sample) for sample in trainset_slide]
            # trainset_slide_norm_array = np.array(trainset_slide_norm)
            
            # prior_size = config['dataloader']['prior_size']
            # trainset_slide_norm = []
            # for sample in trainset_slide:
            #     sample[:prior_size,:] = scaler.fit_transform(sample[:prior_size,:])
            #     # sample[prior_size:,:] = scaler.fit_transform(sample[prior_size:,:])
            #     trainset_slide_norm.append(sample)
                
            # # trainset_slide_norm = [scaler.fit_transform(sample) for sample in trainset_slide]
            
            step_size_eval = config['dataloader']['step_size_eval']
            evalset_slide = sliding_window(evalset, window_size, step_size_eval)
            evalset_slide_norm_array = np.array(evalset_slide)
            
            # evalset_slide_norm = []
            # for sample in evalset_slide:
            #     sample[:prior_size,:] = scaler.fit_transform(sample[:prior_size,:])
            #     # sample[prior_size:,:] = scaler.fit_transform(sample[prior_size:,:])
            #     evalset_slide_norm.append(sample)
            
            step_size_test = config['dataloader']['step_size_test']
            testset_slide = sliding_window(testset, window_size, step_size_test)
            
            for i in range(len(testset_slide)):
                testset_i = testset_slide[i,:]
                testset_i_norm = testset_i.copy()
                # testset_i_norm[:prior_size,:] = scaler.fit_transform(testset_i_norm[:prior_size,:])
                
                test_dict[str(i+1)] = {
                    'norm_slide': testset_i_norm,
                    'ori_slide': testset_i,
                    # 'scaler': scaler
                }

        with h5py.File(config['dataloader']['matrix_path'], 'w') as f_w:
            data_group = f_w.create_group('datas_train_eval')
            data_group.create_dataset('trainset', data=trainset_slide_norm_array)
            data_group.create_dataset('evalset', data=evalset_slide_norm_array)
            
            test_group = f_w.create_group('testset_grp')
            for key, data in test_dict.items():
                key_group = test_group.create_group(key)
                key_group.create_dataset(name='norm_slide', data=data['norm_slide'])
                key_group.create_dataset(name='ori_slide', data=data['ori_slide'])
                
                # scaler_group = key_group.create_group('scaler')
                # scaler_group.create_dataset('min_', data=data['scaler'].min_)
                # scaler_group.create_dataset('scale_', data=data['scaler'].scale_)
                # scaler_group.create_dataset('data_min_', data=data['scaler'].data_min_)
                # scaler_group.create_dataset('data_max_', data=data['scaler'].data_max_)
                # scaler_group.create_dataset('data_range_', data=data['scaler'].data_range_)
                
                # scaler_group.create_dataset('mean_', data=data['scaler'].mean_)
                # scaler_group.create_dataset('std_', data=data['scaler'].var_)
    return


def data_prepare_generate(config):
    if os.path.exists(config['dataloader']['matrix_path']):
        return
    else:
        scaler = MinMaxScaler(feature_range=(-1,1))
        # train_dict, eval_dict = {}, {}
        name_item = 0
        with h5py.File(config['dataloader']['matrix_path'], 'w') as f_w:
            data_grp = f_w.create_group(name='train_datas')
            
            with h5py.File(config['dataloader']['dataset_path'], 'r') as f_r:
                data_group = f_r['datas']
                for key in data_group:
                    dataset_ori = data_group[key] # (n, 6)
                    dataset_norm = scaler.fit_transform(dataset_ori)
                    if dataset_ori.shape[0] < config['dataloader']['window_size']:
                        continue
                    dataset_norm_slide = sliding_window(dataset_norm, config)
                    
                    activity_idx = int(key.split('_')[1].split('a')[1])
                    
                    for i in range(dataset_norm_slide.shape[0]):
                        data_set = data_grp.create_dataset(name=str(name_item), data=dataset_norm_slide[i,:])
                        data_set.attrs['activity'] = activity_idx - 1
                        name_item += 1
                    
                
    return



def data_prepare_generate_uschad(config):
    if os.path.exists(config['dataloader']['matrix_path']):
        return
    else:
        uschad_para_path = './Datas/uschad_activity_parameters.h5'
        activity_para_dict = load_activity_parameters(uschad_para_path)
        
        window_size = config['dataloader']['window_size']
        step_size = config['dataloader']['step_size_train']
        
        name_item = 0
        
        with h5py.File(config['dataloader']['matrix_path'], 'w') as f_w:
            data_grp = f_w.create_group(name='train_datas')
            
            with h5py.File(config['dataloader']['dataset_path'], 'r') as f_r:
                data_group = f_r['datas']
                for key in data_group:
                    dataset_ori = data_group[key] # (n, 6)
                    
                    activity_i = key.split('_')[1].split('a')[1]
                    
                    normalized_data_i = apply_normalization(dataset_ori, activity_i, activity_para_dict)
                    dataset_norm_slide = sliding_window(normalized_data_i, window_size, step_size)
                    
                    for i in range(dataset_norm_slide.shape[0]):
                        data_set = data_grp.create_dataset(name=str(name_item), data=dataset_norm_slide[i,:])
                        data_set.attrs['activity'] = int(activity_i) - 1
                        name_item += 1
                        
    return



def data_prepare_uschad_TimeDiff(data_path_in, data_path_out):

    train_list = [1,2,3,4,5,6,7,8,9]
    eval_list = [10,11]
    test_list = [12,13,14]
    
    window_size = 368
    step_size_train = 16
    step_size_eval = 32
    step_size_test = 32

    train_, eval_, test_ = [],[],[]
    
    uschad_para_path = './Datas/uschad_activity_parameters.h5'
    activity_para_dict = load_activity_parameters(uschad_para_path)
    
    with h5py.File(data_path_in, 'r') as f_r:
        data_group = f_r['datas']
        for key in data_group:
            dataset_ori = data_group[key] # (n, 6)
            
            subject_i = int(key.split('_')[0].split('b')[1])
            activity_i = key.split('_')[1].split('a')[1]
            
            normalized_data_i = apply_normalization(dataset_ori, activity_i, activity_para_dict)
            
            if int(subject_i) in train_list:
                dataset_norm_slide = sliding_window(normalized_data_i, window_size, step_size_train)
                train_.append(dataset_norm_slide)
            elif int(subject_i) in eval_list:
                dataset_norm_slide = sliding_window(normalized_data_i, window_size, step_size_eval)
                eval_.append(dataset_norm_slide)
            elif int(subject_i) in test_list:
                dataset_norm_slide = sliding_window(normalized_data_i, window_size, step_size_test)
                test_.append(dataset_norm_slide)
            else:
                raise 'Error'
                
    train_ = np.vstack(train_)
    eval_ = np.vstack(eval_)
    test_ = np.vstack(test_)
    
    with h5py.File(data_path_out, 'w') as f_w:
        data_group = f_w.create_group('datas')
        data_group.create_dataset('train', data=train_)
        data_group.create_dataset('val', data=eval_)
        data_group.create_dataset('test', data=test_)
    return


if __name__ == '__main__':
    data_path_in = './Datas/uschad_dataset.h5'
    data_path_out = '../master_prediction_v7.0 simplified/datasets/uschad_metrix.h5'
    
    data_prepare_uschad_TimeDiff(data_path_in, data_path_out)


