from class_utils import Distortion_mode, Distortion, Distortion_precompute
from dataset import SavePrecomputedEncodingCityscapesDataset, SavePrecomputedEncodingStackedCityscapesDataset
import torch
import time
import datetime
import os
import math

################################################################################################################
################################################################################################################
################################################################################################################

def initialize_datasets_dataloader(distortion: Distortion, distortion_precompute_mode: Distortion_precompute, num_workers, splits = ['train', 'val', 'test']):
    undistorted_dataset_folder_string = "/data/cityscapes/"
    distorted_dataset_folder_string = f'/data/cityscapes_{distortion}/'
    
    if distortion_precompute_mode == Distortion_precompute.PRECOMPUTED_ENCODING:
        print("Prepare encoding preprocessing of dataset with parameters", distortion, "...")
        datasets = {x : SavePrecomputedEncodingCityscapesDataset(undistorted_dataset_folder_string, distorted_dataset_folder_string, x, distortion) for x in splits}
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=num_workers, shuffle=False, num_workers=num_workers) for x in splits}
    elif distortion_precompute_mode == Distortion_precompute.PRECOMPUTED_ENCODING_STACKED:
        print("Prepare stacked encoding preprocessing computation of dataset with parameters", distortion, "...")
        lossless_distortion = Distortion(Distortion_mode.LOSSLESS, color_subsampling="444" if distortion.color_subsampling == "420" else distortion.color_subsampling, subsampling_factor=distortion.subsampling_factor)
        datasets = {x : SavePrecomputedEncodingStackedCityscapesDataset(f'/data/cityscapes_{lossless_distortion}', f'/data/cityscapes_{distortion}', x, distortion) for x in splits}
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=num_workers, shuffle=False, num_workers=num_workers) for x in splits}
    else:
        raise Exception("Unknown distortion precompute mode value " + str(distortion_precompute_mode))

    return datasets, dataloaders, distorted_dataset_folder_string

################################################################################################################

def run_compression_preprocessing(distortion: Distortion, distortion_precompute_mode: Distortion_precompute, splits = ['train', 'val', 'test']):
    num_workers = 8
    datasets, dataloaders, folder_string = initialize_datasets_dataloader(distortion, distortion_precompute_mode, num_workers, splits)
    os.makedirs(folder_string, exist_ok=True)
    os.chdir(folder_string)
    try:
        os.symlink('../cityscapes/gtFine/', 'gtFine', target_is_directory=True)
    except FileExistsError:
        print('Tried to create a symlink that already exists. Continuing...')
    
    for phase in splits:
        len_dataloader = len(dataloaders[phase])
        print_density = math.ceil(len_dataloader/10)

        start_time = time.time()
        if phase == 'train':
            print('\n---- TRAIN ----')
        elif phase == 'val':
            print('\n---- VAL ----')
        elif phase == 'test':
            print('\n---- TEST ----')
        
        batch_i = 0
        for bitstreams_length in dataloaders[phase]:
            if batch_i % (5*print_density) == 0:
                print(f'Batch {batch_i} / {len_dataloader}')
            batch_i += 1
            
            if batch_i % print_density == 0:
                batches_left = len_dataloader - (batch_i + 1)
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time) / (batch_i + 1))
                print(f'    ---- ETA {time_left}')
        try:
            os.symlink(f'../cityscapes/{phase}.txt', f'{phase}.txt')
        except FileExistsError:
            print('Tried to create a symlink that already exists. Continuing...')

################################################################################################################
################################################################################################################
################################################################################################################