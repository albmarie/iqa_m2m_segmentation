import sys
sys.path.append("/src")
import compression_preprocessing
import class_utils

if __name__ == "__main__":
    distortion = class_utils.Distortion(class_utils.Distortion_mode.JPEG, quality=20, color_subsampling="420", subsampling_factor=1.0)
    distortion_precompute_mode = class_utils.Distortion_precompute.PRECOMPUTED_ENCODING

    compression_preprocessing.run_compression_preprocessing(distortion, distortion_precompute_mode, splits = ['val'])