from PIL import Image
import numpy as np
import io
import os
import random
import string

####################################################################################################

# Another ffmpeg version (4.3) is installed via conda in the docker image pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
# Using the absolute path for the ffmpeg bin ensure that the version explicitly installed in the Dockerfile (6.1) is used
ffmpeg_bin_path = '/ffmpeg/bin/ffmpeg' 

# Run the bash command cmd. If hide_output is set to True, output of cmd command won't be printed to the standard output. 
def run_bash_cmd(cmd, hide_output=True):
    if hide_output:
        cmd = "(" + cmd + ")> /dev/null 2>&1"
    os.system(cmd)

####################################################################################################
##########          TRANSFORMATIONS          #######################################################
####################################################################################################

"""
Perform a PNG compression.

Require a RGB PIL image, return a RGB PIL image.
"""
class PNG(object):
    def __init__(self, color_subsampling = '444', p: float = 1.0):
        self.bitstream_format=".png"
        assert color_subsampling == '444' or color_subsampling == '400' #color_subsampling == '400': ITU-R 601-2 (L = R * 299/1000 + G * 587/1000 + B * 114/1000)
        self.color_subsampling = color_subsampling
        self.p = p
        self.len = -1
        self.codec = "PNG"

    def encoder(self, sample, image_path):
        if self.color_subsampling == '400':
            sample.convert('L').save(image_path + self.bitstream_format, self.codec, optimize=True)
        elif self.color_subsampling == '444':
            sample.save(image_path + self.bitstream_format, self.codec, optimize=True)
        else:
            raise Exception(f'Unknown value for self.color_subsampling ({self.color_subsampling}). Valid values are \'444\' and \'400\'.')
        return os.stat(image_path + self.bitstream_format).st_size
    
    def decoder(self, image_path):
        return Image.open(image_path + self.bitstream_format)
    
    def free_memory(self, image_path, keep_bitstream):
        if not keep_bitstream:
            rm_cmd = "rm " + image_path + self.bitstream_format
            run_bash_cmd(rm_cmd)

    def __call__(self, sample):
        rand = np.random.uniform()
        perform_transform = rand <= self.p
        if not perform_transform:
            self.len = 0
            return sample

        buffer = io.BytesIO()
        if self.color_subsampling == '400':
            sample.convert('L').save(buffer, self.codec, optimize=True)
        elif self.color_subsampling == '444':
            sample.save(buffer, self.codec, optimize=True)
        else:
            raise Exception(f'Unknown value for self.color_subsampling ({self.color_subsampling}). Valid values are \'444\' and \'400\'.')
        self.len = buffer.getbuffer().nbytes
        return Image.open(buffer)

####################################################################################################

"""
Perform a JPEG compression with the provided quality parameter with a probability p.

Require a RGB PIL image, return a RGB PIL image.
"""
class JPEG(object):
    def __init__(self, quality: int, color_subsampling = "420", optimize = False, p: float = 1.0):
        self.quality = quality
        self.color_subsampling = color_subsampling
        self.optimize = optimize
        self.bitstream_format=".jpg"
        self.p = p
        self.len = -1
        self.codec = "JPEG"

    def encoder(self, sample, image_path):
        if self.color_subsampling == "400": #Greyscale
            sample.convert('L').save(image_path + self.bitstream_format, self.codec, quality=self.quality, optimize=self.optimize)
        else:
            sample.save(image_path + self.bitstream_format, self.codec, quality=self.quality, subsampling=self.color_subsampling[0] + ":" + self.color_subsampling[1] + ":" + self.color_subsampling[2], optimize=self.optimize)
        return os.stat(image_path + self.bitstream_format).st_size

    def decoder(self, image_path):
        return Image.open(image_path + self.bitstream_format).convert('RGB')
    
    def free_memory(self, image_path, keep_bitstream):
        if not keep_bitstream:
            rm_cmd = "rm " + image_path + self.bitstream_format
            run_bash_cmd(rm_cmd)

    def __call__(self, sample):
        rand = np.random.uniform()
        perform_transform = rand <= self.p
        if not perform_transform:
            self.len = 0
            return sample

        buffer = io.BytesIO()
        if self.color_subsampling == "400": #Greyscale
            sample.convert('L').save(buffer, self.codec, quality=self.quality, optimize=self.optimize)
        else:
            sample.save(buffer, self.codec, quality=self.quality, subsampling=self.color_subsampling[0] + ":" + self.color_subsampling[1] + ":" + self.color_subsampling[2], optimize=self.optimize)
        self.len = buffer.getbuffer().nbytes
        return Image.open(buffer).convert('RGB')

####################################################################################################

"""
Perform a compression with VVenC (https://github.com/fraunhoferhhi/vvenc) with the provided qP parameter with a probability p.

Require a RGB PIL image, return a RGB PIL image.
"""
class VVenC(object):
    def __init__(self, qP: int, color_subsampling = "420", p: float = 1.0):
        self.qP = qP
        self.color_subsampling = color_subsampling
        #"yuv420p" if color_subsampling == "400" means that image without chroma is stored in 420 yuv format (sub optimal but easier for compatibility)
        self.ffmpeg_color_subsampling = "yuv420p" if color_subsampling == "420" else "yuv420p" if color_subsampling == "400" else None
        if self.ffmpeg_color_subsampling is None:
            raise Exception("Invalid color_subsampling " + color_subsampling)
        self.in_format=".ppm"
        self.bitstream_format=".266"
        self.out_format=".ppm"
        self.p = p
        self.len = -1
        self.codec = "VVENC"

    def encoder(self, sample, image_path, get_cmd=False):
        sample_width, sample_height = sample.size
        enc_cmd  = f"{ffmpeg_bin_path} -y -i {image_path}{self.in_format} -pix_fmt {self.ffmpeg_color_subsampling} {image_path}.yuv && "
        enc_cmd += f"/vvenc/bin/release-static/vvencFFapp -fr 1 -g 1 --InputBitDepth 8 --InternalBitDepth 8 -t 0 --preset fast --ALFSpeed 0 -qpa \
            -s {sample_width}x{sample_height} -q {self.qP} -i {image_path}.yuv -b {image_path}{self.bitstream_format} && "
        enc_cmd += f"rm {image_path}{self.in_format} {image_path}.yuv"

        if get_cmd:
            return enc_cmd
        
        if self.color_subsampling == "400": #Greyscale
            sample.convert('L').save(image_path + self.in_format)
        else:
            sample.save(image_path + self.in_format)
        run_bash_cmd(enc_cmd)
        self.len = os.stat(image_path + self.bitstream_format).st_size

    def decoder(self, image_path, sample_width, sample_height, get_cmd=False):
        dec_cmd  = f"/vvdec/bin/release-static/vvdecapp -t 0 -v 0 -b {image_path}{self.bitstream_format} -o {image_path}_rec.yuv && "
        dec_cmd += f"{ffmpeg_bin_path} -y -f rawvideo -video_size {sample_width}x{sample_height} -pix_fmt {self.ffmpeg_color_subsampling} -i {image_path}_rec.yuv {image_path}{self.out_format} && "
        dec_cmd += f"rm {image_path}_rec.yuv"

        if get_cmd:
            return dec_cmd

        run_bash_cmd(dec_cmd)
        #print("VVenC Done!", end=" ", flush=True)
        return Image.open(image_path + self.out_format).convert("RGB")

    def free_memory(self, image_path, keep_bitstream):
        rm_cmd  = "rm " + image_path + self.out_format
        rm_cmd += "" if keep_bitstream else (" && rm " + image_path + self.bitstream_format)
        run_bash_cmd(rm_cmd)
       
    def __call__(self, sample):
        rand = np.random.uniform()
        perform_transform = rand <= self.p
        if not perform_transform:
            self.len = 0
            return sample
        
        image_path = "/mnt/ram_partition/_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
        self.encoder(sample, image_path)
        compressed_image = self.decoder(image_path, sample.size[0], sample.size[1]) #maybe height and width are inverted (no issue for now since used image have same width and height)
        self.free_memory(image_path, keep_bitstream=True)

        return compressed_image

####################################################################################################

"""
Perform a compression with x265 (using ffmpeg) with the provided qP parameter with a probability p.
x265 compression is done with the use of ffmpeg.

Require a RGB PIL image, return a RGB PIL image.
"""
class x265(object):
    def __init__(self, qP: int, color_subsampling = "420", p: float = 1.0):
        self.qP = qP
        self.color_subsampling = color_subsampling
        #"yuv420p" if color_subsampling == "400" means that image without chroma is stored in 420 yuv format (sub optimal but easier for compatibility)
        self.ffmpeg_color_subsampling = "yuv420p" if color_subsampling == "420" else "yuv420p" if color_subsampling == "400" else None
        if self.ffmpeg_color_subsampling is None:
            raise Exception("Invalid color_subsampling " + color_subsampling)
        self.in_format=".ppm"
        self.bitstream_format=".265"
        self.out_format=".ppm"
        self.p = p
        self.len = -1
        self.codec = "x265"

    def encoder(self, sample, image_path, get_cmd=False):
        sample_width, sample_height = sample.size
        enc_cmd  = f"{ffmpeg_bin_path} -y -i {image_path}{self.in_format} -pix_fmt {self.ffmpeg_color_subsampling} {image_path}.yuv && "
        enc_cmd += f"{ffmpeg_bin_path} -y -f rawvideo -video_size {sample_width}x{sample_height} -i {image_path}.yuv \
            -c:v libx265 -preset slow -x265-params qp={self.qP}:amp=1:rskip=0:no-limit-modes=1:no-info=1:fps=1:keyint=1:ref=1:no-open-gop=1:weightp=0:weightb=0:cutree=0:rc-lookahead=0:bframes=0:scenecut=0:b-adapt=0:repeat-headers=1:rdoq-level=2:psy-rdoq=0:psy-rd=0\
            {image_path}{self.bitstream_format} && "
        enc_cmd += f"rm {image_path}{self.in_format} {image_path}.yuv"

        if get_cmd:
            return enc_cmd
        
        if self.color_subsampling == "400": #Greyscale
            sample.convert('L').save(image_path + self.in_format)
        else:
            sample.save(image_path + self.in_format)
        run_bash_cmd(enc_cmd)
        self.len = os.stat(image_path + self.bitstream_format).st_size

    def decoder(self, image_path, sample_width, sample_height, get_cmd=False):
        dec_cmd  = f" {ffmpeg_bin_path} -y -i {image_path}{self.bitstream_format} -f rawvideo -video_size {sample_width}x{sample_height} -pix_fmt {self.ffmpeg_color_subsampling} {image_path}_rec.yuv && "
        dec_cmd += f"{ffmpeg_bin_path} -y -f rawvideo -video_size {sample_width}x{sample_height} -pix_fmt {self.ffmpeg_color_subsampling} -i {image_path}_rec.yuv {image_path}{self.out_format} && "
        dec_cmd += f"rm {image_path}_rec.yuv"

        if get_cmd:
            return dec_cmd

        run_bash_cmd(dec_cmd)
        return Image.open(image_path + self.out_format).convert("RGB")

    def free_memory(self, image_path, keep_bitstream):
        rm_cmd  = "rm " + image_path + self.out_format
        rm_cmd += "" if keep_bitstream else (" && rm " + image_path + self.bitstream_format)
        run_bash_cmd(rm_cmd)
       
    def __call__(self, sample):
        rand = np.random.uniform()
        perform_transform = rand <= self.p
        if not perform_transform:
            self.len = 0
            return sample
        
        image_path = "/mnt/ram_partition/_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
        self.encoder(sample, image_path)
        compressed_image = self.decoder(image_path, sample.size[0], sample.size[1]) #maybe height and width are inverted (no issue for now since used image have same width and height)
        self.free_memory(image_path, keep_bitstream=True)

        return compressed_image

####################################################################################################

"""
Perform a JM compression (https://vcgit.hhi.fraunhofer.de/jvet/JM/-/tree/master) with the provided qP parameter with a probability p.

Require a RGB PIL image, return a RGB PIL image.
"""
class JM(object):
    def __init__(self, qP: int, color_subsampling = "420", p: float = 1.0):
        self.qP = qP
        self.color_subsampling = color_subsampling
        #"yuv420p" if color_subsampling == "400" means that image without chroma is stored in 420 yuv format (sub optimal but easier for compatibility)
        self.ffmpeg_color_subsampling = "yuv420p" if color_subsampling == "420" else "yuv420p" if color_subsampling == "400" else None
        if self.ffmpeg_color_subsampling is None:
            raise Exception("Invalid color_subsampling " + color_subsampling)
        self.in_format=".ppm"
        self.bitstream_format=".264"
        self.out_format=".ppm"
        self.p = p
        self.len = -1
        self.codec = "JM"
    
    def encoder(self, sample, image_path, get_cmd=False):
        cfg_file = "/tmp/JM_preprocessed_image_properties_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=16)) + ".cfg"

        run_bash_cmd(f'cp /src/JM_preprocessed_image_properties.cfg {cfg_file}')
        tmp_str = image_path.replace("\"", "").replace("\'", "")
        with open(cfg_file, 'a') as f:
            f.write(f'\nInputFile = \"{tmp_str}.yuv\"')
            f.write(f'\nOutputFile = \"{tmp_str}{self.bitstream_format}\"') 
        
        source_width, source_height = sample.size[0], sample.size[1]
        enc_cmd  = f"{ffmpeg_bin_path} -y -i {image_path}{self.in_format} -pix_fmt {self.ffmpeg_color_subsampling} {image_path}.yuv && " 
        enc_cmd += f"/JM/bin/lencod.exe -d {cfg_file} -p FramesToBeEncoded=1 -p LevelIDC=42 -p SourceWidth={source_width} -p SourceHeight={source_height} -p QPISlice={self.qP} || " #This command will fail to generate logs but will generate .264 bitstream. This is done on purpose (avoid writing/deleting useless file on disk)
        enc_cmd += f"rm {image_path}{self.in_format} {image_path}.yuv {cfg_file}"        

        if get_cmd:
            return enc_cmd

        if self.color_subsampling == "400": #Greyscale
            sample.convert('L').save(image_path + self.in_format)
        else:
            sample.save(image_path + self.in_format)
        run_bash_cmd(enc_cmd)
        self.len = os.stat(image_path + self.bitstream_format).st_size

    def decoder(self, image_path, sample_width, sample_height, get_cmd=False):
        cfg_file = "/tmp/JM_preprocessed_image_properties_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=16)) + ".cfg"

        run_bash_cmd(f'cp /src/JM_preprocessed_image_properties.cfg {cfg_file}')
        tmp_str = image_path.replace("\"", "").replace("\'", "")
        with open(cfg_file, 'a') as f:
            f.write(f'\nInputFile = \"{tmp_str}{self.bitstream_format}\"')
            f.write(f'\nOutputFile = \"{tmp_str}_rec.yuv\"')
        
        image_path_dirname = os.path.dirname(image_path)
        dec_cmd  = f"/JM/bin/ldecod.exe -d {cfg_file} && "
        dec_cmd += f"{ffmpeg_bin_path} -y -f rawvideo -video_size {sample_width}x{sample_height} -pix_fmt {self.ffmpeg_color_subsampling} -i {image_path}_rec.yuv {image_path}{self.out_format} && "
        dec_cmd += f"rm {image_path}_rec.yuv {image_path_dirname}log.dec {image_path_dirname}dataDec.txt {cfg_file}"

        if get_cmd:
            return dec_cmd

        run_bash_cmd(dec_cmd)
        return Image.open(image_path + self.out_format).convert("RGB")

    def free_memory(self, image_path, keep_bitstream):
        rm_cmd  = "rm " + image_path + self.out_format
        rm_cmd += "" if keep_bitstream else (" && rm " + image_path + self.bitstream_format)
        run_bash_cmd(rm_cmd)
        
    def __call__(self, sample):
        rand = np.random.uniform()
        perform_transform = rand <= self.p
        if not perform_transform:
            self.len = 0
            return sample
        
        image_path = "/mnt/ram_partition/_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
        self.encoder(sample, image_path)
        compressed_image = self.decoder(image_path, sample.size[0], sample.size[1]) #maybe height and width are inverted (no issue for now since used image have same width and height)
        self.free_memory(image_path, keep_bitstream=True)

        return compressed_image

####################################################################################################

class RGB2Gray(object):
    def __init__(self, single_channel = True):
        self.single_channel = single_channel

    def __call__(self, sample):
        if self.single_channel:
            return sample.convert('L')
        else:
            return sample.convert('L').convert('RGB') #Convert back to RGB to keep 3 channels

####################################################################################################

class Resize(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, sample):
        return sample.resize(self.img_size, resample=Image.BICUBIC)

####################################################################################################

class toFloat(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample.float()

####################################################################################################
####################################################################################################
####################################################################################################