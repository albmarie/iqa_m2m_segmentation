from enum import Enum
import transforms as T

################################################################################################################
##########          CLASS UTILS          #######################################################################
################################################################################################################

class Metric_mode(Enum):
    PSNR    =  0
    SSIM    =  1
    MS_SSIM =  2
    VIFp    =  3
    FSIM    =  4
    SR_SIM  =  5
    GMSD    =  6
    MS_GMSD =  7
    VSI     =  8
    DSS     =  9
    HaarPSI = 10
    MDSI    = 11
    LPIPS   = 12
    PieAPP  = 13
    DISTS   = 14

################################################################################################################

class Correlation_mode(Enum):
    PLCC  = 0
    SROC  = 1
    KROCC = 2

################################################################################################################

class Machine_Perception_mode(Enum):
    acc  = 0
    mIoU = 1

################################################################################################################

class Distortion_mode(Enum):
    LOSSLESS = 0
    JPEG     = 1
    JM       = 2
    x265     = 3
    VVENC    = 4

################################################################################################################

class Distortion(object):
    def __init__(self, distortion_mode: int, quality = None, color_subsampling = None, subsampling_factor: float = None):
        if distortion_mode == Distortion_mode.LOSSLESS:
            if color_subsampling == None:
                color_subsampling = '444'
        
        self.DISTORTION_MODE             = distortion_mode
        self.DISTORTION_STRING           = distortion_mode.name
        if distortion_mode == Distortion_mode.LOSSLESS:
            self.QUALITY_SUPPORTED           = False
        elif distortion_mode == Distortion_mode.JPEG or distortion_mode == Distortion_mode.JM or distortion_mode == Distortion_mode.x265 or distortion_mode == Distortion_mode.VVENC:
            self.QUALITY_SUPPORTED           = True
        else:
            raise Exception("Unsupported distortion mode (" + str(distortion_mode) + ").")
        
        if quality is not None and not self.QUALITY_SUPPORTED:
            raise Exception("Quality not supported with distortion mode (" + str(distortion_mode) + "), but a quality parameter was provided.")
        
        self.quality            = quality
        self.color_subsampling  = color_subsampling
        self.subsampling_factor = subsampling_factor

    #Downsampling part of the whole coding scheme (downsampling -> compression -> decompression -> upsampling). Note that this function could be an upsampling if self.subsampling_factor>1, which is unlikely.
    def get_downsampling_transform(self):
        img_size = (round(2048*self.subsampling_factor), round(1024*self.subsampling_factor))
        return T.Resize(img_size = img_size)

    def get_compress_transform(self):
        if   self.DISTORTION_MODE == Distortion_mode.LOSSLESS:
            return T.PNG(                color_subsampling=self.color_subsampling)
        elif self.DISTORTION_MODE == Distortion_mode.JPEG:
            return T.JPEG( self.quality, color_subsampling=self.color_subsampling, p=1.0)
        elif self.DISTORTION_MODE == Distortion_mode.JM:
            return T.JM(   self.quality, color_subsampling=self.color_subsampling, p=1.0)
        elif self.DISTORTION_MODE == Distortion_mode.x265:
            return T.x265( self.quality, color_subsampling=self.color_subsampling, p=1.0)
        elif self.DISTORTION_MODE == Distortion_mode.VVENC:
            return T.VVenC(self.quality, color_subsampling=self.color_subsampling, p=1.0)
        else:
            raise Exception("Unsupported distortion mode (" + str(self.DISTORTION_MODE) + ").")

    def __str__(self):
        quality_string            = ("quality="           + str(self.quality)            + ", ") if self.QUALITY_SUPPORTED           else ""
        color_subsampling_string  = ("color_subsampling=" + self.color_subsampling       + ", ")
        subsampling_factor_string = "subsampling_factor=" + str(self.subsampling_factor)
        return self.DISTORTION_STRING + "(" + quality_string + color_subsampling_string + subsampling_factor_string + ")"

    def to_list(self):
        return [self.DISTORTION_MODE.value, self.quality, self.color_subsampling, self.subsampling_factor]

################################################################################################################

class Distortion_precompute(Enum):
    PRECOMPUTED_ENCODING          = 0 #Mode to load bitstream saved on disk (decoding only on-the-fly, "low" storage space required) ("low" -> depends on the codec/quality/etc. selected)
    PRECOMPUTED_ENCODING_STACKED  = 1 #Mode to load bitstream saved on disk. Similar to PRECOMPUTED_ENCODING, except that encoded images are regrouped in video to decode multiple image per command

################################################################################################################
################################################################################################################
################################################################################################################