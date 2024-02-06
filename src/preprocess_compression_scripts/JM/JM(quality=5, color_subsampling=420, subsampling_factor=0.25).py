import sys
sys.path.append("/src")
import compression_preprocessing
import class_utils

if __name__ == "__main__":
    distortion = class_utils.Distortion(class_utils.Distortion_mode.JM, quality=5, color_subsampling="420", subsampling_factor=0.25)
    distortion_precompute_mode = class_utils.Distortion_precompute.PRECOMPUTED_ENCODING_STACKED

    compression_preprocessing.run_compression_preprocessing(distortion, distortion_precompute_mode, splits = ['val'])