import torchvision
from transforms import RGB2Gray, toFloat, run_bash_cmd
from torch.utils.data import Dataset
from class_utils import Distortion_mode, Machine_Perception_mode, Distortion, Distortion_precompute
from PIL import Image
import numpy as np
import json
import random
from scipy.ndimage import convolve
import cv2 as cv
from machine_perception_measure import Machine_Perception_fn
import re
import os

################################################################################################################
################################################################################################################
################################################################################################################

transforms_metric = torchvision.transforms.Compose([
    RGB2Gray(single_channel=False),
    torchvision.transforms.PILToTensor(),
    toFloat(),
    torchvision.transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0])])

################################################################################################################
################################################################################################################
################################################################################################################

def get_default_distortion_precompute_mode(distortion: Distortion): 
    if distortion.DISTORTION_MODE == Distortion_mode.LOSSLESS:
        return Distortion_precompute.PRECOMPUTED_ENCODING
    elif distortion.DISTORTION_MODE == Distortion_mode.JPEG:
        return Distortion_precompute.PRECOMPUTED_ENCODING
    elif distortion.DISTORTION_MODE == Distortion_mode.JM:
        return Distortion_precompute.PRECOMPUTED_ENCODING_STACKED
    elif distortion.DISTORTION_MODE == Distortion_mode.x265:
        return Distortion_precompute.PRECOMPUTED_ENCODING_STACKED
    elif distortion.DISTORTION_MODE == Distortion_mode.VVENC:
        return Distortion_precompute.PRECOMPUTED_ENCODING_STACKED
    else:
        raise Exception("Unsupported distortion mode (" + str(distortion.DISTORTION_MODE) + ").")

################################################################################################################

"""
img_path do no contain image extension (e.g. '.png', '.jpg', '.x265')
"""
def load_image_based_on_distortion(img_path: str, distortion: Distortion, img_size = (2048, 1024)):
    distortion_precompute = get_default_distortion_precompute_mode(distortion)

    if distortion_precompute == Distortion_precompute.PRECOMPUTED_ENCODING:
        img_extension = distortion.get_compress_transform().bitstream_format
        return Image.open(f'{img_path}{img_extension}') #img_extension is either .jpg or .png based on get_default_distortion_precompute_mode()
    elif distortion_precompute == Distortion_precompute.PRECOMPUTED_ENCODING_STACKED:
        yuv_img_nb_bytes = int((img_size[0]*img_size[1])*(distortion.subsampling_factor**2)*3/2)
        img_city = img_path.split('/')[-2]
        img_name = f'file \'' + img_path.split('/')[-1] + '.png\''
        with open(f'/data/_stacked_images_order/{img_city}.txt', "r") as stack_img_idx_names_file:
            stack_img_idx_names = stack_img_idx_names_file.read().split("\n")[:-1]
        with open('/'.join(img_path.split('/')[:-1]) + f'/{img_city}_rec.yuv', "br") as yuv_file:
            if img_name not in stack_img_idx_names:
                raise Exception("Reached a part of the code that should not be reachable. Aborting...")
            yuv_file.seek(stack_img_idx_names.index(img_name)*yuv_img_nb_bytes)
            data = yuv_file.read(yuv_img_nb_bytes)
            return Image.fromarray(cv.cvtColor(np.frombuffer(data, dtype=np.uint8).reshape((int(img_size[1]*distortion.subsampling_factor*1.5), int(img_size[0]*distortion.subsampling_factor))), cv.COLOR_YUV2RGB_I420))
    else:
        raise Exception(f'Unsupported distortion precompute mode {distortion_precompute}.')

################################################################################################################

"""
Return a CTU coordinates of CTU_size containing the whole CU of coords CU_coords and size CU_size.
It is checked that returned coordinates refer to a block that is inside the image.
"""
def find_matching_CTU_coords(CU_coords, CU_size, CTU_size, img_size):
    c = 0
    while True:
        CU_coords_in_CTU = (CU_size[0]*random.randrange(int(CTU_size/CU_size[0])), CU_size[1]*random.randrange(int(CTU_size/CU_size[1])))
        CTU_coords = (CU_coords[0]-CU_coords_in_CTU[0], CU_coords[1]-CU_coords_in_CTU[1])
        if CTU_coords[0] >= 0 and CTU_coords[1] >= 0 and CTU_coords[0]+CTU_size <= img_size[0] and CTU_coords[1]+CTU_size <= img_size[1]:
            return CTU_coords
        c = c+1

################################################################################################################

#TODO: REPLACE BY NEW CLASS IN machine_perception_measure.py
"""
def compute_acc(distorted_prediction_img_path, pseudo_ground_truth_img_path, CU_coords, CU_size):
    pseudo_ground_truth_img = Image.open(pseudo_ground_truth_img_path).convert('L')
    distorted_prediction_img = Image.open(distorted_prediction_img_path).convert('L')

    if CU_coords is not None:
        pseudo_ground_truth_img = pseudo_ground_truth_img.crop((CU_coords[0], CU_coords[1], CU_coords[0]+CU_size[0], CU_coords[1]+CU_size[1]))
        distorted_prediction_img = distorted_prediction_img.crop((CU_coords[0], CU_coords[1], CU_coords[0]+CU_size[0], CU_coords[1]+CU_size[1]))
    
    if CU_size is None:
        CU_size = (2048, 1024) #Default image size
    return (CU_size[0]*CU_size[1] - len(np.nonzero(np.array(pseudo_ground_truth_img, dtype=np.float32) != np.array(distorted_prediction_img, dtype=np.float32))[0]))/(CU_size[0]*CU_size[1])
"""

################################################################################################################

def load_block_dict(idx, split, img_id, distortion, CU_coords, CTU_size, block_size, block_acc):
    block_dict = {}
    block_dict['idx'] = idx
    block_dict['acc'] = block_acc
    block_dict['CU_coords'] = (CU_coords[0], CU_coords[1])

    tmp_img = Image.open(f'/data/cityscapes/leftImg8bit/{split}/{img_id}_leftImg8bit.png')
    undistorted_CU = tmp_img.crop((CU_coords[0], CU_coords[1], CU_coords[0]+block_size[0], CU_coords[1]+block_size[1]))
    block_dict['undistorted_CU_metric'] = transforms_metric(undistorted_CU)

    tmp_img = load_image_based_on_distortion(f'/data/cityscapes_{distortion}/leftImg8bit/{split}/{img_id}_leftImg8bit', distortion)
    tmp_img = tmp_img if distortion.subsampling_factor == 1.0 else tmp_img.resize((2048, 1024), resample=Image.BICUBIC)
    distorted_CU = tmp_img.crop((CU_coords[0], CU_coords[1], CU_coords[0]+block_size[0], CU_coords[1]+block_size[1]))
    block_dict['distorted_CU_metric'] = transforms_metric(distorted_CU)

    return block_dict

################################################################################################################

def load_image_dict(idx, split, img_id, distortion, machine_perception_fn):
    image_dict = {}
    image_dict['idx'] = idx
    image_dict['distortion'] = f'{distortion}'

    image_dict['undistorted_img'] = transforms_metric(Image.open(f'/data/cityscapes/leftImg8bit/{split}/{img_id}_leftImg8bit.png'))

    tmp_img = load_image_based_on_distortion(f'/data/cityscapes_{distortion}/leftImg8bit/{split}/{img_id}_leftImg8bit', distortion)
    tmp_img = tmp_img if distortion.subsampling_factor == 1.0 else tmp_img.resize((2048, 1024), resample=Image.BICUBIC)
    image_dict['distorted_img'] = transforms_metric(tmp_img)

    image_dict[f'{machine_perception_fn.machine_perception.name}'] = machine_perception_fn(distorted_prediction_img_path = f'/prediction_folder/cityscapes_{distortion}/{split}/{img_id}_leftImg8bit.png',
                                                                                      pseudo_ground_truth_img_path = f'/pseudo_ground_truth_folder/{split}/{img_id}_leftImg8bit.png',
                                                                                      CU_coords = None,
                                                                                      CU_size = None)

    return image_dict

################################################################################################################
################################################################################################################
################################################################################################################

"""
Block sampling algorithm proposed in section III.E of the QoMEX 2023 paper, where the accuracy of each sampled block is uniformly distributed in [0, 1]. 

For each random block that is selected, we make sure the corresponding accuracy is not almost always the same (e.g. 1).

To do so, for each block, we draw a random number i uniformly in [0,k[.
For example, if k=4 and i=2, the valid number of correctly classified pixels for the block must be in [0.5, 0.75].

While the number of correctly classified pixels of the current block is not in between [i/k, (i+1)/k], we reject the current block and try another one.
To avoid infinite loops, the algorithm described in section III.E of the QoMEX 2023 paper will check if there is at least one valid block coordinate in the current image where the resulting accuracy is between [i/k, (i+1)/k].
If there is at least one valid block candidate, one of these block coordinates is selected as the sampled block.
If there is no valid block candidate within the current image and the current value of i, the algorithm repeats with a new image till a valid block coordinate is found.
"""
class CustomCityscapesBlockNormalizedRandomDataset(Dataset):
    def __init__(self, return_loaded_blocks:bool, coding_config_list, split: str, block_size_list, pdf_list, img_size = (2048, 1024), num_sample: int = 5000, cities = None, k: int = 10):
        self.return_loaded_blocks = return_loaded_blocks
        with open(f'/data/cityscapes/{split}.txt', "r") as f:
            split_images_path = f.read()
        self.split_images_path = []
        split_images_path = split_images_path.split("\n")[:-1]
        if cities is not None:
            for city in cities:
                self.split_images_path.extend(list(filter(lambda img_id: city in img_id, split_images_path)))
        
        self.block_size_list, self.pdf_list = block_size_list, pdf_list
        self.CTU_size = 128
        self.update_block_size()
        self.coding_config_list = coding_config_list
        self.split = split
        self.img_size = img_size
        self.num_sample = num_sample
        self.k = k

    def update_block_size(self):
        self.current_CU_size = random.choices(self.block_size_list, weights=self.pdf_list, k=1)[0]

    def __len__(self):
        if self.num_sample is None:
            return len(self.split_images_path)
        else:
            return self.num_sample

    def __getitem__(self, idx):
        i_acc = random.randrange(self.k)
        
        continue_while_loop = True
        while continue_while_loop:
            if self.num_sample is None:
                img_id = self.split_images_path[idx]
            else:
                random_img_idx = random.randrange(len(self.split_images_path))
                img_id = self.split_images_path[random_img_idx]
            random_distortion_idx = random.randrange(len(self.coding_config_list))
            random_distortion = self.coding_config_list[random_distortion_idx]
            
            pseudo_ground_truth_img = Image.open(f'/pseudo_ground_truth_folder/{self.split}/{img_id}_leftImg8bit.png')
            distorted_prediction_img = Image.open(f'/prediction_folder/cityscapes_{random_distortion}/{self.split}/{img_id}_leftImg8bit.png')

            misclassified_pixels_coords = np.nonzero(np.array(pseudo_ground_truth_img, dtype=np.float32) != np.array(distorted_prediction_img, dtype=np.float32))[:2]
            misclassified_pixels_img = np.ones((pseudo_ground_truth_img.size[1], pseudo_ground_truth_img.size[0]), dtype=np.float32)
            for pix_idx in range(len(misclassified_pixels_coords[0])):
                misclassified_pixels_img[misclassified_pixels_coords[0][pix_idx]][misclassified_pixels_coords[1][pix_idx]] = 0
            #Two 1D convolution instead of one 2D convolution (complexity from n**2 to 2n for each pixel)
            misclassified_neighbor_img = convolve(convolve(misclassified_pixels_img, np.ones((self.current_CU_size[1], 1), dtype=np.uint8), mode='constant', cval=float('nan')),
                                                    np.ones((1, self.current_CU_size[0]), dtype=np.uint8), mode='constant', cval=float('nan'))
            
            block_num_pix = self.current_CU_size[0]*self.current_CU_size[1]
            if not self.split == 'test':
                ground_truth_mask = cv.cvtColor(np.array(Image.open(f'/data/cityscapes/gtFine/{self.split}/{img_id}_gtFine_color.png')), cv.COLOR_RGB2GRAY)
                ground_truth_mask = ground_truth_mask.astype(np.float32)
                ground_truth_mask[ground_truth_mask==0] = np.nan
            if not i_acc == self.k-1:
                valid_block_coords = np.nonzero((misclassified_neighbor_img>=(i_acc/self.k)*block_num_pix) & (misclassified_neighbor_img<((i_acc+1)/self.k)*block_num_pix))
            else:
                valid_block_coords = np.nonzero((misclassified_neighbor_img>=(i_acc/self.k)*block_num_pix) & (misclassified_neighbor_img<=((i_acc+1)/self.k)*block_num_pix))
            for random_coords_idx in random.sample(range(len(valid_block_coords[0])), len(valid_block_coords[0])):
                CU_coords = (valid_block_coords[1][random_coords_idx]-int((self.current_CU_size[0]-1)/2), valid_block_coords[0][random_coords_idx]-int((self.current_CU_size[1]-1)/2))
                if not self.split == 'test':
                    tmp = ground_truth_mask[CU_coords[1]:CU_coords[1]+self.current_CU_size[0], CU_coords[0]:CU_coords[0]+self.current_CU_size[1]]
                    if tmp.sum() == 0.0:
                        print("coords: ",  CU_coords[0], CU_coords[0]+self.current_CU_size[1],CU_coords[1], CU_coords[1]+self.current_CU_size[0])
                        print(ground_truth_mask.shape)
                        print(tmp.shape)
                    if not np.isnan(tmp).any():
                        continue_while_loop = False
                        break
                else:
                    continue_while_loop = False
                    break
        
        random_img_id = self.split_images_path[random_img_idx]
        block_acc = misclassified_neighbor_img[valid_block_coords[0][random_coords_idx]][valid_block_coords[1][random_coords_idx]]/block_num_pix

        if not self.return_loaded_blocks:
            return random_img_id, random_distortion_idx, CU_coords, self.current_CU_size, block_acc
        else:
            return load_block_dict(idx, self.split, random_img_id, self.coding_config_list[random_distortion_idx], CU_coords, self.CTU_size, self.current_CU_size, block_acc)

################################################################################################################

"""
Load blocks based on the provided base_json_path.
This base_json_path specify which block should be sampled by providing the img_id, the distortion, the block coordinates and the block size (see function load_block_dict()).
"""
class LoadCustomCityscapesBlockFixedRandomDataset(Dataset):
    def __init__(self, base_json_path: str, split: str):
        val_blocks_json_path = f'{base_json_path}_val_blocks.json'
        with open(val_blocks_json_path, 'r') as json_file:
            self.block_dict = json.load(json_file)
        self.CTU_size = 128
        self.split = split
        
    def __len__(self):
        return len(self.block_dict)

    def __getitem__(self, idx):
        img_id, distortion_list, CU_coords, block_size, block_acc = self.block_dict[f'{idx}']
        
        distortion = Distortion(distortion_mode=Distortion_mode(distortion_list[0]),
                                quality=distortion_list[1],
                                color_subsampling=distortion_list[2],
                                subsampling_factor=distortion_list[3])
        
        block_dict = load_block_dict(idx, self.split, img_id, distortion, CU_coords, self.CTU_size, block_size, block_acc)
        return block_dict

################################################################################################################

class CustomCityscapesDataset(Dataset):
    def __init__(self, coding_config_list, machine_perception_mode: Machine_Perception_mode, split: str, num_sample = None, cities = None):
        with open(f'/data/cityscapes/{split}.txt', "r") as f:
            split_images_path = f.read()
        self.split_images_path = []
        split_images_path = split_images_path.split("\n")[:-1]
        if cities is not None:
            for city in cities:
                self.split_images_path.extend(list(filter(lambda img_id: city in img_id, split_images_path)))
        
        self.coding_config_list = coding_config_list
        self.split = split
        self.num_sample = num_sample


        self.machine_perception_fn = Machine_Perception_fn(machine_perception_mode)

    def __len__(self):
        if self.num_sample is None:
            return len(self.split_images_path)*len(self.coding_config_list)
        else:
            return self.num_sample
    
    def __getitem__(self, idx):
        if self.num_sample is None:
            img_id = self.split_images_path[idx%len(self.split_images_path)]
            random_distortion = self.coding_config_list[idx//len(self.split_images_path)]
        else:
            img_id = self.split_images_path[random.randrange(len(self.split_images_path))]
            random_distortion = self.coding_config_list[random.randrange(len(self.coding_config_list))]

        return load_image_dict(idx, self.split, img_id, random_distortion, self.machine_perception_fn)

################################################################################################################
##########          PREPROCESSING DATASETS          ############################################################
################################################################################################################

"""
Encode images from the Cityscapes dataset using the coding configuration distortion and save the corresponding bitstreams of the disk.
This dataset class is associated with the enum Distortion_precompute.PRECOMPUTED_ENCODING.
"""
class SavePrecomputedEncodingCityscapesDataset(torchvision.datasets.Cityscapes):
    def __init__(self, undistorted_root: str, distorted_root: str, split: str, distortion: Distortion) -> None:
        if not(distortion.DISTORTION_MODE == Distortion_mode.LOSSLESS or distortion.DISTORTION_MODE == Distortion_mode.JPEG):
            raise Exception("Unsupported distortion " + distortion.DISTORTION_STRING + ". Supported distortions are: LOSSLESS, JPEG, JPEG2000.")
        self.undistorted_root = undistorted_root
        self.distorted_root = distorted_root
        
        self.compress_transform = distortion.get_compress_transform()
        
        self.extension_discriminator = re.compile(r'\.png')
        super(SavePrecomputedEncodingCityscapesDataset, self).__init__(undistorted_root, split=split, mode="fine", target_type="semantic", transform=distortion.get_downsampling_transform())
 
    def __getitem__(self, idx):
        downsampled_sample, _ = torchvision.datasets.Cityscapes.__getitem__(self, idx)
        
        distorted_sample_filepath = self.images[idx]
        distorted_sample_filepath = distorted_sample_filepath.replace(self.undistorted_root, self.distorted_root, 1)
        os.makedirs(os.path.dirname(distorted_sample_filepath), exist_ok=True)

        bitstream_length = self.compress_transform.encoder(downsampled_sample, distorted_sample_filepath.replace(self.extension_discriminator.search(distorted_sample_filepath).group(0), ""))
        self.compress_transform.free_memory(distorted_sample_filepath.replace(self.extension_discriminator.search(distorted_sample_filepath).group(0), ""), keep_bitstream=True)

        return bitstream_length

################################################################################################################

"""
Encode images from the Cityscapes dataset using the coding configuration distortion and save the corresponding bitstreams of the disk.
This dataset class is associated with the enum Distortion_precompute.PRECOMPUTED_ENCODING_STACKED.
The stacked keyword means that images with a folder of the cityscapes dataset (e.g. lindau) are regrouped into a single YUV video file to be encoded together, where each frame of the video represent one of the original images.
This is useful to encode all images with fewer calls as some encoders are meant for videos and not still images (i.e. JM, x265 and VVenC).
All-intra configuration is used for these video encoders.

Each __item__ of the stacked dataset represents a video of multiple images (all images that belongs to a given city).
This stacked dataset have a number of __item__ correspoding to the number of folder in the split (e.g. 3 for validation set as the validation is composed of 3 cities: lindau, munster and frankfurt).
"""
class SavePrecomputedEncodingStackedCityscapesDataset(Dataset):
    def __init__(self, lossless_preprocessed_downsampled_root: str, distorted_root: str, split: str, distortion: Distortion):
        if not(distortion.DISTORTION_MODE == Distortion_mode.JM or distortion.DISTORTION_MODE == Distortion_mode.x265 or distortion.DISTORTION_MODE == Distortion_mode.VVENC):
            raise Exception("Unsupported distortion " + distortion.DISTORTION_STRING + ". Supported distortions are JM, x265 and VVenC.")
        self.lossless_preprocessed_downsampled_root = lossless_preprocessed_downsampled_root
        self.distorted_root = distorted_root
        self.split = split

        self.distortion = distortion
        self.compress_transform = distortion.get_compress_transform()

        self.folders = []
        for (dirpath, _, filenames) in os.walk(os.path.join(self.lossless_preprocessed_downsampled_root, 'leftImg8bit/', self.split)):
            if len(filenames) < 1:
                continue
            self.folders.append(os.path.basename(dirpath))

    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, idx):
        folder = self.folders[idx]
        distorted_folder = os.path.join(self.distorted_root, "leftImg8bit/", self.split, folder)
        undistorted_folder = os.path.join(self.lossless_preprocessed_downsampled_root, "leftImg8bit/", self.split, folder)
        os.makedirs(distorted_folder, exist_ok=True)

        filenames = None
        for (_, _, _filenames) in os.walk(undistorted_folder):
            if len(_filenames) > 0:
                filenames = _filenames
                break

        if f'{folder}.txt' in filenames:
            filenames.remove(f'{folder}.txt')
        
        distorted_folder = "\"" + distorted_folder + "\""
        undistorted_folder = "\"" + undistorted_folder + "\""

        stacked_images_order_dest_file = folder + ".txt"

        enc_cmd = self.compress_transform.encoder(sample=Image.new(mode="RGB", size=(round(2048*self.distortion.subsampling_factor), round(1024*self.distortion.subsampling_factor))),
                                                image_path=os.path.join(distorted_folder, stacked_images_order_dest_file[:-4]),
                                                get_cmd=True)
        enc_cmd = enc_cmd.replace("-i " + os.path.join(distorted_folder, folder) + self.compress_transform.in_format,
                                  "-r 1 -f concat -i " + undistorted_folder + "/" + folder + ".txt", 1)
        enc_cmd = enc_cmd.replace(" rm " + os.path.join(distorted_folder, folder) + self.compress_transform.in_format + " &&", "")
        if self.distortion.DISTORTION_MODE == Distortion_mode.JM:
            enc_cmd = enc_cmd.replace("FramesToBeEncoded=1 ", "FramesToBeEncoded=" + str(len(filenames)) + " ")

        pre_enc_cmd = "cp /stacked_images_order/" + folder + ".txt " + undistorted_folder + "/" + stacked_images_order_dest_file + " && cd " + undistorted_folder + " && "
        
        print("pre_enc_cmd + enc_cmd", pre_enc_cmd + enc_cmd)
        run_bash_cmd(pre_enc_cmd + enc_cmd, hide_output=False)
        return os.stat(distorted_folder[1:-1] + "/" + folder + self.compress_transform.bitstream_format).st_size/len(filenames)

################################################################################################################
################################################################################################################
################################################################################################################