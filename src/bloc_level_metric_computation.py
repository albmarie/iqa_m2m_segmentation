from dataset import CustomCityscapesBlockNormalizedRandomDataset, LoadCustomCityscapesBlockFixedRandomDataset
import torch
from tqdm import tqdm
import json
from metrics import Metric_fn
from correlations import Correlation_fn
from datetime import datetime
import random
import string
from terminaltables import AsciiTable
import numpy as np
from PIL import Image
from class_utils import Correlation_mode, Metric_mode, Distortion_mode, Distortion
import matplotlib
matplotlib.rcParams['pdf.fonttype'], matplotlib.rcParams['ps.fonttype'] = 42, 42 #Avoid Type3 fonts
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################################################

"""
Sample a total of num_sample block following the block sampling algorithm proposed in section III.E of the QoMEX 2023 paper.
Data required to sample back the blocks sampled by the block sampling algorithm (img_id, distortion, block coordinates and block size) is written on the disk in a json file.
Parameter coding_config_list sepcify which distortion is considered (see main for an example where the #C = 4*4*11 = 176 coding configuration are considered).
"""
def save_random_val_blocks_json(base_json_path: str, coding_config_list, block_size: int, num_sample: int, split:str, cities = None):
    
    dataset = CustomCityscapesBlockNormalizedRandomDataset(return_loaded_blocks=False,
                                                            coding_config_list=coding_config_list,
                                                            split=split,
                                                            block_size_list=[block_size],
                                                            pdf_list=[1.0],
                                                            num_sample=num_sample,
                                                            cities=cities,
                                                            k=10)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=18)
    sampled_block_dict = {}
    for sample_idx, (random_img_id, random_distortion_idx, random_coords, block_size, block_acc) in enumerate(tqdm(dataloader)):
        sampled_block_dict[sample_idx] = [random_img_id[0],
                                            coding_config_list[random_distortion_idx.item()].to_list(),
                                            [random_coords[0].item(), random_coords[1].item()],
                                            [block_size[0].item(), block_size[1].item()],
                                            block_acc.item()]

    val_blocks_json_path = f'{base_json_path}_val_blocks.json'
    with open(val_blocks_json_path, 'w') as json_file:
        json.dump(sampled_block_dict, json_file)

################################################################################################################

"""
Load json file saved on disk by the function save_random_val_blocks_json() to sample back blocks.
The function save_FR_IQA_metrics_val_scores() aim to compute FR IQA metric scores between undistorted and distorted blocks.
Considered FR IQA metrics depend on the passed metrics parameter.
Save a new json file on disk, so that computed FR IQA metric can be loaded back.
"""
def save_FR_IQA_metrics_val_scores(base_json_path: str, metrics, split: str, batch_size: int = 1, num_workers: int = 0):
    hardware = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(hardware)

    dataset = LoadCustomCityscapesBlockFixedRandomDataset(base_json_path, split=split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    metrics_fn_dict = {}
    for metric in metrics:
        metrics_fn_dict[metric.name] = Metric_fn(metric, device=device)

    sampled_block_dict = {}
    for _, (batch_dict) in enumerate(tqdm(dataloader)):
        batch_undistorted, batch_distorted = batch_dict['undistorted_CU_metric'], batch_dict['distorted_CU_metric'] 
        batch_idx, batch_acc = batch_dict['idx'], batch_dict['acc']

        batch_distorted, batch_undistorted = batch_distorted.to(device), batch_undistorted.to(device)
        tmp_metrics_values_dict = {}
        for metric in metrics:
            tmp_metrics_values_dict[metric.name] = metrics_fn_dict[metric.name](batch_distorted, batch_undistorted)
        
        for idx, idx_batch in enumerate(batch_idx):
            sampled_block_dict[f'{idx_batch}'] = {}
            sampled_block_dict[f'{idx_batch}']['acc'] = batch_acc[idx].item()
            for metric in metrics:
                sampled_block_dict[f'{idx_batch}'][metric.name] = tmp_metrics_values_dict[metric.name][idx].item()
    metrics_json_path = f'{base_json_path}_val_metrics.json'
    with open(metrics_json_path, 'w') as json_file:
        json.dump(sampled_block_dict, json_file)

################################################################################################################

"""
Load json files saved on disk by the functions save_random_val_blocks_json() and save_FR_IQA_metrics_val_scores().
Compute correlation between pre-computed FR IQA metrics scores and semantic segmentation pixel-wise accuracy.
The parameter correlation_modes allow to specify by which way correlation should be computed (e.g. PLCC, SROC and/or KROCC).
Write a json file on disk with computed correlation values.
"""
def save_FR_IQA_val_correlation_scores(base_json_path: str, metrics, correlation_modes):
    metrics_json_path = f'{base_json_path}_val_metrics.json'
    correlation_json_path = f'{base_json_path}_val_correlation.json'
    
    with open(metrics_json_path, 'r') as json_file:
        metrics_val_dict = json.load(json_file)
    
    acc_scores, conventional_metric_scores = [], {}
    for metric in metrics:
        conventional_metric_scores[metric.name] = []

    for idx in range(len(metrics_val_dict)):
        acc_scores.append(metrics_val_dict[f'{idx}']['acc'])
        for metric in metrics:
            conventional_metric_scores[metric.name].append(metrics_val_dict[f'{idx}'][metric.name])

    correlation_dict = {}
    for metric in tqdm(metrics):
        correlation_dict[metric.name] = {}
        for correlation_mode in correlation_modes:
            correlation_dict[metric.name][correlation_mode.name] = Correlation_fn(correlation_mode)(acc_scores, conventional_metric_scores[metric.name])

    with open(correlation_json_path, 'w') as json_file:
        json.dump(correlation_dict, json_file)

################################################################################################################

"""
Call in succession save_random_val_blocks_json(), save_FR_IQA_metrics_val_scores() and save_FR_IQA_val_correlation_scores() to compute correlation between FR IQA metric scores and semantic segmentation pixel-wise accuracy.
Passed parameter fixed_config is a dictionary that allow to control various hyper-parameters (e.g. block size on which correlation should be computed).
See main() function for an example
"""
def precompute_val_multiscale_blocks(coding_config_list, metrics, fixed_config):
    time_now = datetime.now()
    fixed_config['base_json_path'] = '/src/val_config/' + time_now.strftime("%Y-%m-%d-%H-%M-%S") + '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))

    base_json_path = fixed_config['base_json_path']
    for block_size in fixed_config['block_size_list']:
        print(f'Precomputing val epoch for block size {block_size[0]}x{block_size[1]}...')
        block_width, block_height = block_size
        current_base_json_path = f'{base_json_path}_{block_width}x{block_height}'
        save_random_val_blocks_json(current_base_json_path, coding_config_list, block_size, num_sample=2**12, split=fixed_config['cities_split'], cities=fixed_config['cities_eval'])
        save_FR_IQA_metrics_val_scores(current_base_json_path, metrics, split=fixed_config['cities_split'], batch_size=2**6, num_workers=fixed_config['nb_cpu'])
        save_FR_IQA_val_correlation_scores(current_base_json_path, metrics, fixed_config['correlation_mode'])

################################################################################################################

def print_correlation_table(fixed_config):
    correlation_tables = None
    
    for block_size in fixed_config['block_size_list']:
        base_json_path = fixed_config['base_json_path']
        with open(f'{base_json_path}_{block_size[0]}x{block_size[1]}_val_correlation.json', 'r') as json_file:
            correlation_val_dict = json.load(json_file)
        
        correlation_table = [[f'{block_size[0]}x{block_size[1]}']]
        for correlation_mode in fixed_config['correlation_mode']:
            correlation_table[0].append(correlation_mode.name)
        for metric in metrics:
            correlation_table_line = [metric.name]
            for correlation_mode in fixed_config['correlation_mode']:
                correlation_score = correlation_val_dict[metric.name][correlation_mode.name]
                if correlation_score == correlation_score: #If tmp is not float('NaN')...
                    correlation_table_line.append("%.4f" % abs(correlation_score[0]))
                else:
                    correlation_table_line.append("%.4f" % abs(correlation_score))
            correlation_table.append(correlation_table_line)

        if correlation_tables is None:
            correlation_tables = correlation_table
        else:
            correlation_tables = [x+y for x,y in zip(correlation_tables, correlation_table)]
    
    print(AsciiTable(correlation_tables).table)

################################################################################################################

def save_distribution_plot_acc(fixed_config, fig_size_multiplier:int = 1):
    base_json_path  = fixed_config['base_json_path']

    for block_size in fixed_config['block_size_list']:
        metrics_json_path = f'{base_json_path}_{block_size[0]}x{block_size[1]}_val_blocks.json'

        with open(metrics_json_path, 'r') as json_file:
            val_blocks_dict = json.load(json_file)

        acc_list = []
        for idx in range(len(val_blocks_dict)):
            _, _, _, _, _, bloc_acc = val_blocks_dict[f'{idx}']
            acc_list.append(bloc_acc)
        
        fig, ax = plt.subplots(figsize=(6*fig_size_multiplier, 5*fig_size_multiplier))
        plt.subplots_adjust(top = 0.975,
                            bottom = 0.17,
                            right = 0.985,
                            left = 0.175,
                            hspace = 0,
                            wspace = 1)
        sns.histplot(data=acc_list, stat='probability', ax=ax, binwidth=0.1)
        ax.set_xlabel(xlabel = 'Accuracy')
        ax.set_ylim(bottom = 0.0, top = 1.0)
        fig.savefig(f'{base_json_path}_{block_size[0]}x{block_size[1]}_acc_probability_map.pdf')    
    
################################################################################################################

def save_probality_maps(fixed_config):
    img_size = (1024, 2048)

    for block_size in fixed_config['block_size_list']:
        base_json_path = fixed_config['base_json_path']
        with open(f'{base_json_path}_{block_size[0]}x{block_size[1]}_val_blocks.json', 'r') as json_file:
            val_blocks_dict = json.load(json_file)
        probability_map = np.zeros(img_size, dtype=np.uint16)
        for sample in val_blocks_dict:
            _, _, (coords_w, coords_h), (block_w, block_h), _ = val_blocks_dict[sample]
            probability_map[coords_h:coords_h+block_h, coords_w:coords_w+block_w] += 1
        
        probability_map_img = (255.*probability_map/np.max(probability_map)).astype(np.uint8)
        Image.fromarray(probability_map_img).save(f'{base_json_path}_{block_size[0]}x{block_size[1]}_probability_map.png')

################################################################################################################

def save_metrics_acc_scatter_plots(fixed_config, metrics, fig_size_multiplier:int = 1):
    base_json_path  = fixed_config['base_json_path']
    sns.set_theme()

    for block_size in fixed_config['block_size_list']:
        metrics_json_path = f'{base_json_path}_{block_size[0]}x{block_size[1]}_val_metrics.json'

        with open(metrics_json_path, 'r') as json_file:
            metrics_val_dict = json.load(json_file)

        acc_list = []
        for idx in range(len(metrics_val_dict)):
            acc_list.append(metrics_val_dict[f'{idx}']['acc'])

        for metric in metrics:
            metric_list = []
            for idx in range(len(metrics_val_dict)):
                metric_list.append(metrics_val_dict[f'{idx}'][metric.name])
            
            fig, ax = plt.subplots(figsize = (6*fig_size_multiplier, 5*fig_size_multiplier))
            plt.subplots_adjust(top = 0.975,
                                bottom = 0.21,
                                right = 0.98,
                                left = 0.2,
                                hspace = 0,
                                wspace = 1)
            ax.scatter(acc_list, metric_list, marker = "s", s=0.1*matplotlib.rcParams['lines.markersize'], label=f'{block_size[0]}x{block_size[1]}')
            ax.set_xlabel(xlabel = 'Accuracy')
            ax.set_ylabel(ylabel = f'{metric.name}')
            ax.set_ylim(bottom = 0.0, top = 0.75)
            fig.savefig(f'{base_json_path}_{block_size[0]}x{block_size[1]}_scatter_plot_{metric.name}.pdf')


################################################################################################################
##########          MAIN          ##############################################################################
################################################################################################################

if __name__ == "__main__":
    coding_config_list = []

    for subsampling_factor in [0.25, 0.5, 0.75, 1.0]:
        for quality in [90, 70, 50, 35, 30, 25, 20, 15, 10, 7, 5]:
            coding_config_list.append(Distortion(Distortion_mode.JPEG, quality=quality , color_subsampling="420", subsampling_factor=subsampling_factor))
        for quality in [5*i for i in list(range(11))]: #[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            coding_config_list.append(Distortion(Distortion_mode.JM   , quality=quality, color_subsampling="420", subsampling_factor=subsampling_factor))
            coding_config_list.append(Distortion(Distortion_mode.x265 , quality=quality, color_subsampling="420", subsampling_factor=subsampling_factor))
            coding_config_list.append(Distortion(Distortion_mode.VVENC, quality=quality, color_subsampling="420", subsampling_factor=subsampling_factor))

    fixed_config = {
        'correlation_mode': [Correlation_mode.PLCC, Correlation_mode.SROC, Correlation_mode.KROCC],
        'block_size_list': [(32,32), (64,64), (128,128)],
        'base_json_path': '/src/val_config/2023-02-10-13-13-27_JXWAL0C7OJ5X5OW1',
        'nb_cpu': 8,
        #'cities_split': 'test',
        #'cities_eval': ['berlin', 'mainz', 'munich', 'bielefeld', 'bonn', 'leverkusen'],
        'cities_split': 'val',
        'cities_eval': ['frankfurt', 'lindau', 'munster'],
    }

    metrics = [Metric_mode.PSNR,
                Metric_mode.SSIM,
                Metric_mode.MS_SSIM,
                Metric_mode.FSIM,
                Metric_mode.SR_SIM,
                Metric_mode.GMSD,
                Metric_mode.MS_GMSD,
                Metric_mode.VSI,
                Metric_mode.HaarPSI,
                Metric_mode.MDSI,
                Metric_mode.LPIPS,
                Metric_mode.DISTS]
    
    #precompute_val_multiscale_blocks(coding_config_list, metrics, fixed_config)
    print_correlation_table(fixed_config)
    #save_distribution_plot_acc(fixed_config, fig_size_multiplier=0.5)
    #save_probality_maps(fixed_config)
    #save_metrics_acc_scatter_plots(fixed_config, metrics, fig_size_multiplier=0.5)

################################################################################################################