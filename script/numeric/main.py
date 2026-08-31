# >----------------------------------------------------------------------------------------------------
# > Data Driven Multiscale Simulation (DDMS)
# > This project aims to create a preprocessing/postprocessing tools 
#   for abaqus, damask numerical simulations
# > @Author: Kyle Chien
# > @Github: https://github.com/KyleChien/DDMS_private
# >----------------------------------------------------------------------------------------------------
import os
import json
import shutil
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from ddms.numeric import HandlerDamask
from ddms.numeric.processing import get_loadsteps_RW

logging.basicConfig(level=logging.INFO, format='%(message)s')

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / 'config'

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputyaml', type=str, required=True, dest='yaml')
parser.add_argument('-m', '--mode', type=str, default='test', dest='mode')
parser.add_argument('-tn', '--taskname', type=str, default='', dest='taskname')
parser.add_argument('-t', '--target', type=str, default='', dest='target')
parser.add_argument('-rd', '--ROOTdir', type=str, default='', dest='ROOTdir')
parser.add_argument('--dream3d-runner', type=str, default=os.environ.get('DREAM3D_RUNNER', ''),
					 dest='dream3d_runner',
					 help='path to DREAM3D PipelineRunner executable (mode=genPipeline only); '
						  'defaults to the DREAM3D_RUNNER environment variable')
args = parser.parse_args()

yaml = str(CONFIG_DIR / f'{args.yaml}.yaml')

if __name__=='__main__':
	# DAMASK preprocessing 
	if args.mode=='pre':
		handler = HandlerDamask(yaml, args.taskname, args.target, args.ROOTdir)
		for _ in handler.params_updater():
			handler.c_params()
			handler.g_params()
			handler.preprocessing()

	# generate RVEs with different texture
	elif args.mode=='genPipeline':
		handler = HandlerDamask(yaml, args.taskname, args.target, args.ROOTdir)

		if not args.dream3d_runner:
			parser.error('genPipeline mode requires --dream3d-runner (or the DREAM3D_RUNNER env var) '
						 'to point at your local DREAM3D PipelineRunner executable')

		with open(CONFIG_DIR / 'pipelineMorient20.json', 'r') as f:
			pipeline = json.load(f)
		pRunner = args.dream3d_runner

		# textures = ['singleCrystal', 'biCrystal', 'random', 'rolled']
		textures = ['morient_20']
		for n, texture in enumerate(textures):
			shift = 1
			num_sample = 1
			n_start = num_sample*n + shift
			n_end = n_start+num_sample
			handler.createPipeline(pipeline, texture, n_start, n_end)
			handler.runPipeline(pRunner, texture, n_start, n_end)

	# generate RVEs with same texutre but different loading 
	elif args.mode=='genSameRVE':
		handler = HandlerDamask(yaml, args.taskname, args.target, args.ROOTdir)

		core_name = 'morient_20_16_smooth'
		template_name = f'{core_name}_501'
		start_idx = int(template_name.split('_')[-1]) + 1
		end_idx = start_idx + 499
		for i in tqdm(range(start_idx, end_idx)):
			src_path = os.path.join(handler.prm.ROOTdir, template_name)
			dst_path = os.path.join(handler.prm.ROOTdir, f'{core_name}_{i}')
			shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
			handler.prm.WORKdir = dst_path
			handler.getLoad(loadsteps=get_loadsteps_RW(save_root=handler.prm.WORKdir))

	# generate training dataset from DAMASK result
	elif args.mode=='genDataset':
		handler = HandlerDamask(yaml, args.taskname, args.target, args.ROOTdir)
		handler.update_params()
		handler.createGraphDataset()
	
	# debug
	else:
		handler = HandlerDamask(yaml, args.taskname, args.target, args.ROOTdir)
		for _ in handler.params_updater():
			handler.c_params()
			handler.g_params()


