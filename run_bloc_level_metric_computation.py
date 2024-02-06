import argparse
import os
import subprocess
import getpass

####################################################################################################
####################################################################################################
####################################################################################################

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--data", help="<REQUIRED> Path to folder containing datasets.", type=str, required=True)
	parser.add_argument("-pg", "--pseudo_ground_truth_folder", help="<REQUIRED> Path to folder containing images considered as ground truth.", type=str, required=True)
	parser.add_argument("-p", "--prediction_folder", help="<REQUIRED> Path to folder containing predictions of AI vision task on distorted images.", type=str, required=True)
	parser.add_argument('--gpu', help="Allow GPU utilisation (by default, False).", dest='gpu', action='store_true')
	parser.add_argument('--singularity', help="Run a command using singularity instead of docker (by default, False).", dest='singularity', action='store_true')
	parser.add_argument('--no_cache_link', help="Link cache folder used by pytorch to not download model weights at every execution (by default, True)", dest='no_cache_link', action='store_true')
	opt = parser.parse_args()
	
	script_folder = os.path.dirname(os.path.abspath(__file__))
	if not opt.singularity:
		os.system("sudo docker build -t iqa_m2m_segmentation:latest --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) $(dirname ${0})")
		print("#" * 50)
    
	user = getpass.getuser()
	cmd = []
	if opt.singularity:
		cmd.extend(['singularity', 'run'])
		if opt.gpu:
			cmd.extend(['--nv'])
		cmd.extend(['--bind', os.path.abspath(opt.data) + ':/data/'])
		cmd.extend(['--bind', os.path.dirname(opt.pseudo_ground_truth_folder) + '/:/pseudo_ground_truth_folder/'])
		cmd.extend(['--bind', os.path.dirname(opt.prediction_folder) + '/:/prediction_folder/'])
		cmd.extend(['--bind', script_folder + '/src/:/src/'])
		if not opt.no_cache_link:
			cmd.extend(['--bind', f'/home/{user}/.cache/torch/hub/checkpoints/:/home/user/.cache/torch/hub/checkpoints/'])
		cmd.extend([script_folder + '/singularity/iqa_m2m_segmentation.sif'])
	else:
		cmd.extend(['sudo', 'docker', 'run', '-it'])
		if opt.gpu:
			cmd.extend(['--shm-size=32g', '--gpus', 'all', '--rm'])
		cmd.extend(['-v', os.path.abspath(opt.data) + ':/data/'])
		cmd.extend(['-v', os.path.dirname(opt.pseudo_ground_truth_folder) + '/:/pseudo_ground_truth_folder/'])
		cmd.extend(['-v', os.path.dirname(opt.prediction_folder) + '/:/prediction_folder/'])
		cmd.extend(['-v', script_folder + '/src/:/src/'])
		if not opt.no_cache_link:
			cmd.extend(['-v', f'/home/{user}/.cache/torch/hub/:/home/user/.cache/torch/hub/'])
		cmd.extend(['iqa_m2m_segmentation:latest'])
	cmd.extend(['/bin/bash', '-c'])
	sub_cmd  = 'cd /src/ && python3 /src/bloc_level_metric_computation.py'
	cmd.append(sub_cmd)

	subprocess.run(cmd)

####################################################################################################
####################################################################################################
####################################################################################################

if __name__ == "__main__":
    main()