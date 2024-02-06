import argparse
import os
import subprocess

####################################################################################################
####################################################################################################
####################################################################################################

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--data", help="<REQUIRED> Path to folder containing datasets (the base one)", type=str, required=True)
	parser.add_argument('--singularity', help="Run a command using singularity instead of docker (by default, False)", dest='singularity', action='store_true')
	parser.add_argument("-s", "--script", help="script to run (default, segmentation_dataset_preprocessing.py).", type=str, default="preprocess_compression_scripts/JPEG/JPEG(quality=10, color_subsampling=420, subsampling_factor=1.0).py", required=False)
	opt = parser.parse_args()

	script_folder = os.path.dirname(os.path.abspath(__file__))
	if not opt.singularity:
		os.system("sudo docker build -t iqa_m2m_segmentation:latest --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) $(dirname ${0})")
		print("#" * 50)

	cmd = []
	if opt.singularity:
		cmd.extend(['singularity', 'run'])
		cmd.extend(['--bind', os.path.abspath(opt.data) + ':/data/'])
		cmd.extend(['--bind', script_folder + '/src/:/src/'])
		cmd.extend(['--bind', script_folder + '/stacked_images_order/:/stacked_images_order/'])
		cmd.extend([script_folder + '/singularity/iqa_m2m_segmentation.sif'])
	else:
		cmd.extend(['sudo', 'docker', 'run', '-it'])
		cmd.extend(['-v', os.path.abspath(opt.data) + ':/data/'])
		cmd.extend(['-v', script_folder + '/src/:/src/'])
		cmd.extend(['-v', script_folder + '/stacked_images_order/:/stacked_images_order/'])
		cmd.extend(['iqa_m2m_segmentation:latest'])
	cmd.extend(['/bin/bash', '-c'])
	sub_cmd = f'cd /src/ && python3 \'{opt.script}\''
	cmd.append(sub_cmd)

	subprocess.run(cmd)

####################################################################################################
####################################################################################################
####################################################################################################

if __name__ == "__main__":
    main()