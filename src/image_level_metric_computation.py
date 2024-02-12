from dataset import CustomCityscapesDataset
import torch
import torchvision
from tqdm import tqdm
import json
from metrics import Metric_fn
from correlations import Correlation_fn
from datetime import datetime
import random
import string
from terminaltables import AsciiTable
from class_utils import Correlation_mode, Machine_Perception_mode, Metric_mode, Distortion_mode, Distortion

################################################################################################################

"""
Compute correlation between FR IQA metric scores and measure of machine perception.
"""
def precompute_correlation_table(coding_config_list, metrics, fixed_config):
    time_now = datetime.now()
    fixed_config['base_json_path'] = '/src/val_config/' + time_now.strftime("%Y-%m-%d-%H-%M-%S") + '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
    base_json_path = fixed_config['base_json_path'] + '_image'

    hardware = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(hardware)
   
    dataset = CustomCityscapesDataset(coding_config_list,
                                        machine_perception_mode=fixed_config['machine_perception_mode'],
                                        split=fixed_config['cities_split'],
                                        num_sample=fixed_config['num_sample'],
                                        cities=fixed_config['cities_eval'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=fixed_config['batch_size'], shuffle=False, num_workers=fixed_config['nb_cpu'])

    metrics_fn_dict = {}
    for metric in metrics:
        metrics_fn_dict[metric.name] = Metric_fn(metric, device=device)
    
    metrics_val_dict = {}
    for _, (img_dict) in enumerate(tqdm(dataloader)):
        batch_undistorted, batch_distorted = img_dict['undistorted_img'], img_dict['distorted_img']
        batch_idx, batch_machine_perception = img_dict['idx'], img_dict[fixed_config['machine_perception_mode'].name]

        batch_distorted, batch_undistorted = batch_distorted.to(device), batch_undistorted.to(device)
        tmp_metrics_values_dict = {}
        for metric in metrics:
            tmp_metrics_values_dict[metric.name] = metrics_fn_dict[metric.name](batch_distorted, batch_undistorted)
        
        for idx, idx_batch in enumerate(batch_idx):
            metrics_val_dict[f'{idx_batch}'] = {}
            metrics_val_dict[f'{idx_batch}'][fixed_config['machine_perception_mode'].name] = batch_machine_perception[idx].item()
            for metric in metrics:
                metrics_val_dict[f'{idx_batch}'][metric.name] = tmp_metrics_values_dict[metric.name][idx].item()
    
    metrics_json_path = f'{base_json_path}_val_metrics.json'
    with open(metrics_json_path, 'w') as json_file:
        json.dump(metrics_val_dict, json_file)

    machine_perception_scores, conventional_metric_scores = [], {}
    for metric in metrics:
        conventional_metric_scores[metric.name] = []
    
    for idx in range(len(metrics_val_dict)):
        machine_perception_scores.append(metrics_val_dict[f'{idx}'][fixed_config['machine_perception_mode'].name])
        for metric in metrics:
            conventional_metric_scores[metric.name].append(metrics_val_dict[f'{idx}'][metric.name])
    
    #print("machine_perception_scores", machine_perception_scores)
    #print("conventional_metric_scores", conventional_metric_scores)
    
    correlation_dict = {}
    for metric in tqdm(metrics):
        correlation_dict[metric.name] = {}
        for correlation_mode in fixed_config['correlation_mode']:
            correlation_dict[metric.name][correlation_mode.name] = Correlation_fn(correlation_mode)(machine_perception_scores, conventional_metric_scores[metric.name])

    correlation_table = [['']]
    for correlation_mode in fixed_config['correlation_mode']:
        correlation_table[0].append(correlation_mode.name)
    for metric in metrics:
        correlation_table_line = [metric.name]
        for correlation_mode in fixed_config['correlation_mode']:
            correlation_score = correlation_dict[metric.name][correlation_mode.name]
            if correlation_score == correlation_score: #If tmp is not float('NaN')...
                correlation_table_line.append("%.4f" % abs(correlation_score[0]))
            else:
                correlation_table_line.append("%.4f" % abs(correlation_score))
        correlation_table.append(correlation_table_line)

    print(AsciiTable(correlation_table).table)

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
        'machine_perception_mode': Machine_Perception_mode.acc,
        #'machine_perception_mode': Machine_Perception_mode.mIoU, 
        'nb_cpu': 8,
        'batch_size': 1,
        #'cities_split': 'test',
        #'cities_eval': ['berlin', 'mainz', 'munich', 'bielefeld', 'bonn', 'leverkusen'],
        'cities_split': 'val',
        'cities_eval': ['frankfurt', 'lindau', 'munster'],
        'num_sample': None,
    }

    metrics = [Metric_mode.PSNR, Metric_mode.SSIM, Metric_mode.MS_SSIM, Metric_mode.FSIM, Metric_mode.SR_SIM, Metric_mode.GMSD, Metric_mode.MS_GMSD, Metric_mode.VSI, Metric_mode.HaarPSI, Metric_mode.MDSI, Metric_mode.LPIPS, Metric_mode.DISTS]
    
    precompute_correlation_table(coding_config_list, metrics, fixed_config)

################################################################################################################