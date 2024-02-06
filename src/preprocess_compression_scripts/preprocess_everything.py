import sys
sys.path.append("/src")
import compression_preprocessing
import class_utils

if __name__ == "__main__":
    splits = ['val'] # Split considered in the QOMEX2023 paper. 
    #splits = ['train', 'val', 'test'] # Other splits are supported aswell

    for subsampling_factor in [0.25, 0.5, 0.75, 1.0]:
        # Preprocess JPEG compression...
        for quality in [90, 70, 50, 35, 30, 25, 20, 15, 10, 7, 5]: # JPEG qualities
            distortion = class_utils.Distortion(class_utils.Distortion_mode.JPEG, quality=quality, color_subsampling="420", subsampling_factor=subsampling_factor)
            compression_preprocessing.run_compression_preprocessing(distortion, class_utils.Distortion_precompute.PRECOMPUTED_ENCODING, splits=splits)
    
        # Preprocess lossless (used by JM/x265/VVenC encoders)...
        distortion = class_utils.Distortion(class_utils.Distortion_mode.LOSSLESS, color_subsampling="444", subsampling_factor=subsampling_factor)
        compression_preprocessing.run_compression_preprocessing(distortion, class_utils.Distortion_precompute.PRECOMPUTED_ENCODING, splits=splits)

        # Preprocess JM/x265/VVenC compression...
        for codec in [class_utils.Distortion_mode.JM, class_utils.Distortion_mode.x265, class_utils.Distortion_mode.VVENC]:
            for quality in [5*i for i in list(range(11))]: # QP values
                distortion = class_utils.Distortion(codec, quality=quality, color_subsampling="420", subsampling_factor=subsampling_factor)
                compression_preprocessing.run_compression_preprocessing(distortion, class_utils.Distortion_precompute.PRECOMPUTED_ENCODING_STACKED, splits=splits)