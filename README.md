# IQA_M2M_segmentation
Repo for [the paper](https://hal.science/hal-04091521/) accepted at QoMEX2023 and titled "Evaluation of Image Quality Assessment Metrics for Semantic Segmentation in a Machine-to-Machine Communication Scenario".

Get associated data with [this following link](https://zenodo.org/records/10608698?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImY3ZjBiYjIyLWFmODgtNGE4Ny05MmViLTE1NjYzY2FiN2ZmYSIsImRhdGEiOnt9LCJyYW5kb20iOiJhMzE2YzBkNDE0MTZhZTJhMDY5YmUyMDZlZGQ2MDBmOSJ9.aKIJcz3TDFrCD_Ko6Y2Vf3dVepJ1sPbbLPT4hV0dOOq3pfo0uICmyCn7qLKhDBrkjygMOcSXIF6r9PsYFmjaKg). 
## Installation Requirements

With these minimal requirements, all the scripts in this repo can be executed.

- Docker 
- Python 3
- [Argparse](https://pypi.org/project/argparse/) package
- In order to use GPU within docker containers, you may have to install nvidia-container-toolkit and restart docker daemon as instructed [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Get Started
2 scripts are provided to reproduce image-level and block-level correlations as shown in the paper. These scripts are:

### Image-level

To run this experiment, run the script [run_image_level_metric_computation.py](run_image_level_metric_computation.py).
This script will run [image_level_metric_computation.py](src/image_level_metric_computation.py) inside a Docker container.

Note that values for the image-level experiment are not precomputed.
To compute back the values with the paper, see [this section](#recomputing-results-within-the-paper).
GPU can be used to fasten the computation of Full-Reference (FR) Image Quality Assessment (IQA) scores.

### Block-level 

To run this experiment, run the script [run_bloc_level_metric_computation.py](run_bloc_level_metric_computation.py).
This script will run [bloc_level_metric_computation.py](src/bloc_level_metric_computation.py) inside a Docker container.

Note that the block-level experiment imply randomness because of the block sampling algorithm (see Section III-E in [the paper](https://hal.science/hal-04091521/) for more details).
If you want to recompute results from the block-level experiment (see [this section](#recomputing-results-within-the-paper)), you will have slightly different values from the ones in the paper due to randomness.
If you want to obtain back the exact same values as within the paper, json files within the [val_config/](src/val_config/) folder are provided, which contains:
- Position of sampled blocs and in which compressed images blocks should be taken (corresponding files ends with *_val_blocks.json*, for example [this one](src/val_config/2023-02-10-13-13-27_JXWAL0C7OJ5X5OW1_64x64_val_blocks.json) for 64x64 blocs).
- Precomputed scores for each considered FR IQA metric (corresponding files ends with *_val_metrics.json*, for example [this one](src/val_config/2023-02-10-13-13-27_JXWAL0C7OJ5X5OW1_64x64_val_metrics.json) for 64x64 blocs).
- Precomputed correlation scores between machine perception measure and FR IQA scores (corresponding files ends with *_val_correlation.json*, for example [this one](src/val_config/2023-02-10-13-13-27_JXWAL0C7OJ5X5OW1_64x64_val_correlation.json) for 64x64 blocs).


## Recomputing results within the paper

In order to reproduce the results from image-level and block-level experiments, one would need to gather the data as depicted in the below figure.

![image_level](https://github.com/albmarie/iqa_m2m_segmentation/assets/95236596/741535ff-baec-4a20-a9f4-7659c5cf9319)
<p align="center">
Figure 1 from the [QoMEX2023 paper](https://hal.science/hal-04091521/).
</p>

As shown in the left part of the pipeline, 4 type of data are required to reproduce the results:
- Pristine images $I$. These are based on the [Cityscapes dataset](https://www.cityscapes-dataset.com/). Download the archive *leftImg8bit_trainvaltest.zip* and the *gtFine_trainvaltest.zip* archive.
- Compressed images $\hat{I}$. As there is a lot of different encoding configurations (see Section III-A in [the paper](https://hal.science/hal-04091521/) for more details), compressed bitstream are not provided as it is but can be generated back as shown in [this section](#generate-back-compressed-images).
- Prediction on pristine images $P$, referenced as pseudo ground truth in the QoMEX2023 paper. Download the archive *P_GT.zip* with the [following link](https://zenodo.org/records/10608698?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImY3ZjBiYjIyLWFmODgtNGE4Ny05MmViLTE1NjYzY2FiN2ZmYSIsImRhdGEiOnt9LCJyYW5kb20iOiJhMzE2YzBkNDE0MTZhZTJhMDY5YmUyMDZlZGQ2MDBmOSJ9.aKIJcz3TDFrCD_Ko6Y2Vf3dVepJ1sPbbLPT4hV0dOOq3pfo0uICmyCn7qLKhDBrkjygMOcSXIF6r9PsYFmjaKg).
- Prediction on compressed images $\hat{P}$. Download the archive *P_hat_C.zip* with the [following link](https://zenodo.org/records/10608698?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImY3ZjBiYjIyLWFmODgtNGE4Ny05MmViLTE1NjYzY2FiN2ZmYSIsImRhdGEiOnt9LCJyYW5kb20iOiJhMzE2YzBkNDE0MTZhZTJhMDY5YmUyMDZlZGQ2MDBmOSJ9.aKIJcz3TDFrCD_Ko6Y2Vf3dVepJ1sPbbLPT4hV0dOOq3pfo0uICmyCn7qLKhDBrkjygMOcSXIF6r9PsYFmjaKg).

Note that the archives *P_GT.zip* and *P_hat_C.zip* have been generated with a modified version of the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/) library (using DNN weights within the archive *DNN_weights.zip* among other things).

You can find the expected folder structure for these 4 type of data below.

```
├── data/
│   ├── cityscapes/                                                                  # Pristine images
│       ├── leftImg8bit/
│           ├── val/
│               ├── frankfurt/
│                   ├── frankfurt_000000_000294_leftImg8bit.png
│                   ├── ...
│               ├── lindau/
│                   ├── lindau_000000_000019_leftImg8bit.png
│                   ├── ...
│               ├── munster/
│                   ├── munster_000000_000019_leftImg8bit.png
│                   ├── ...
│       ├── gtFine/
│           ├── val/
│               ├── ...
│       ├── val.txt
```

```
├── data/
│   ├── cityscapes_JPEG(quality=90, color_subsampling=420, subsampling_factor=1.0)/  # Compressed images for one of the coding configurations (176 coding configurations in total)
│       ├── leftImg8bit/
│           ├── val/
│               ├── frankfurt/
│                   ├── frankfurt_000000_000294_leftImg8bit.jpg
│                   ├── ...
│               ├── lindau/
│                   ├── lindau_000000_000019_leftImg8bit.jpg
│                   ├── ...
│               ├── munster/
│                   ├── munster_000000_000019_leftImg8bit.jpg
│                   ├── ...
│       ├── gtFine/                                                                  # Symbolic link to ../cityscapes/gtFine/
│       ├── val.txt                                                                  # Symbolic link to ../cityscapes/val.txt
│   ├── cityscapes_JM(quality=0, color_subsampling=420, subsampling_factor=0.25)/    # Compressed images for another coding configuration
│       ├── leftImg8bit/
│           ├── val/
│               ├── frankfurt/
│                   ├── frankfurt.264                                                # For JM, x265 and VVenC encoders, images are stacked together into a video and encoded all at once
│                   ├── frankfurt_rec.yuv
│               ├── lindau/
│                   ├── lindau.264
│                   ├── lindau_rec.yuv
│               ├── munster/
│                   ├── munster.264
│                   ├── munster_rec.yuv
│       ├── gtFine/                                                                  # Symbolic link to ../cityscapes/gtFine/
│       ├── val.txt                                                                  # Symbolic link to ../cityscapes/val.txt
│   ├── ...
```

```
├── P_GT/                                                                            # Folder containing pseudo ground truth images
│   ├── val/
│       ├── frankfurt/
│           ├── frankfurt_000000_000294_leftImg8bit.png
│           ├── ...
│       ├── lindau/
│           ├── lindau_000000_000019_leftImg8bit.png
│           ├── ...
│       ├── munster/
│           ├── munster_000000_000019_leftImg8bit.png
│           ├── ...
```

```
├── P_hat_C/
│   ├── cityscapes_JPEG(quality=90, color_subsampling=420, subsampling_factor=1.0)/  # Folder containing prediction in images encoded with this coding configuration (176 coding configurations in total)
│       ├── val/
│           ├── frankfurt/
│               ├── frankfurt_000000_000294_leftImg8bit.png
│               ├── ...
│           ├── lindau/
│               ├── lindau_000000_000019_leftImg8bit.png
│               ├── ...
│           ├── munster/
│               ├── munster_000000_000019_leftImg8bit.png
│               ├── ...
│   ├── cityscapes_JPEG(quality=90, color_subsampling=420, subsampling_factor=1.0)/
│       ├── val/
│           ├── frankfurt/
│               ├── frankfurt_000000_000294_leftImg8bit.png
│               ├── ...
│           ├── lindau/
│               ├── lindau_000000_000019_leftImg8bit.png
│               ├── ...
│           ├── munster/
│               ├── munster_000000_000019_leftImg8bit.png
│               ├── ...
│   ├── ...
```

## Generate back compressed images

The script [run_compression_preprocessing.py](run_compression_preprocessing.py) can be used to generate back compressed images from uncompressed one, as done in the paper.

```
python3 run_compression_preprocessing.py -d /path/to/folder/data/ -s preprocess_compression_scripts/preprocess_everything.py
```

You need to pass with the parameter `-d` the path to the *data/* folder (see section [above](#recomputing-results-within-the-paper) for more details).
Folder *cityscapes/* containing pristine images need to be in the *data/* folder.
The script will then compress images with one or multiple coding configuration, depending on the script used with the parameter `-s`.

As an example, the following command:

```
python3 run_compression_preprocessing.py -d /path/to/folder/data/ -s preprocess_compression_scripts/JPEG/JPEG\(quality\=10\,\ color_subsampling\=420\,\ subsampling_factor\=0.5\).py
```

will create one *.jpg* image for each image within the cityscapes validation set (500 in total). Each *.jpg* image will be encoded with JPEG, using a quality of 10 and a downsampling from 2048x1024 pixels down to 1024x512 prior to compression.
If you compress images with all the coding configurations considered in the paper, you can pass the script `preprocess_compression_scripts/preprocess_everything.py` to the parameter `-s` as shown above.

Please check the help using `python3 run_compression_preprocessing.py --help` for more details.
