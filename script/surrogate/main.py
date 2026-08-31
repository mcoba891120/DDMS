# >----------------------------------------------------------------------------------------------------
# > Data Driven Multiscale Simulation (DDMS)
# > This project aims to create a training/testing platform 
#   for machine learning based surrogate model of crystal plasticity simulation
# > @Author: Kyle Chien
# > @Github: https://github.com/KyleChien/DDMS_private
# >----------------------------------------------------------------------------------------------------

if __name__=='__main__':
	import argparse
	from ddms.surrogate import Trainer
	from ddms.surrogate.dataset import MemoryDataset_LSTM
	from ddms.surrogate.model import LMSC
	from ddms.surrogate.callback import VizHomogenizedCS6, VizHomogenizedCSdev6

	parser = argparse.ArgumentParser()
	sys_group = parser.add_argument_group('sys_group', 'group of system variables')
	sys_group.add_argument('-m', '--mode', default='debug', type=str, dest='mode', help='train/test/debug mode')
	sys_group.add_argument('-v', '--verbose', default=1, type=int, dest='verbose', help='verbose level, 1 for debug, 0 for only warning or error')
	sys_group.add_argument('--exp_name', type=str, dest='exp_name', help='name of mlflow experiment')
	sys_group.add_argument('--run_name', type=str, dest='run_name', help='name of mlflow run')
	sys_group.add_argument('--run_id', type=str, dest='run_id', default=None, help='mlflow run_id to resume and load pretrain model from')
	sys_group.add_argument('--sweep_id', type=str, default="", dest='sweep_id', help='wandb sweep_id')
	sys_group.add_argument('--save_root', type=str, dest='save_root', help='root path for saving cache')
	sys_group.add_argument('--data_path', type=str, dest='data_path', help='dataset path')
	sys_group.add_argument('--specify-data', nargs='+', default=None, dest='specify_data', help='data names to contained in dataset, can be multiple')
	sys_group.add_argument('--sync', type=str, dest='sync', default='online', help='offline or online, only used with wandb backend')
	sys_group.add_argument('--backend', type=str, dest='backend', default='wandb', help='choose either mlflow or wandb backend')
	prm_group = parser.add_argument_group('prm_group', 'group of train/test parameters')
	prm_group.add_argument('--epochs', type=int, default=10000, dest='epochs', help='epochs used in this run')
	prm_group.add_argument('--batch_size', type=int, default=50, dest='batch_size', help='batch size used in this run')
	prm_group.add_argument('--learning_rate', type=float, default=0.001, dest='learning_rate', help='learning rate used in this run')
	prm_group.add_argument('--device', type=str, default='cuda', dest='device', help='cuda or cpu')
	args = parser.parse_args()

	Dataset = MemoryDataset_LSTM
	Model = LMSC
	
	if args.mode=='train' or args.mode=='debug':
		trainer = Trainer(args)
		trainer.init(Model, Dataset, hyperparams={
			'in_ch': 5, 'out_ch': 5, 
			'hid_ch': 32, 'stt_ch': 32, 'n_layer': 3, 'tilda_alpha': 1e10,
		})
		trainer.fit(viz_callbacks=[VizHomogenizedCS6(), VizHomogenizedCSdev6()])
	
	
	if args.mode=='test':
		trainer = Trainer(args)
		trainer.init(Model, Dataset, train_split=0, shuffle=False)
		trainer.evaluate(viz_callbacks=[VizHomogenizedCS6(), VizHomogenizedCSdev6()])


	# TODO: testing required
	if args.mode=='sweep':
		import wandb
		def sweep():
			trainer = Trainer(args)
			trainer.init(Model, Dataset, hyperparams={
				'in_ch': 5, 'out_ch': 5,
				'hid_ch': wandb.config.hid_ch, 'stt_ch': wandb.config.stt_ch, 
				'n_layer': wandb.config.n_layer, 'tilda_alpha': wandb.config.tilda_alpha,
			})
			trainer.fit(viz_callbacks=[VizHomogenizedCS6(), VizHomogenizedCSdev6()])

		if args.sweep_id:
			sweep_id = args.sweep_id
		else:
			sweep_config = {
				'method': 'grid', 
				'metric': {
					'goal': 'minimize',
					'name': 'val/loss', 
				},  
				'parameters': {
					'n_layer': {'values': [3]}, 
					'hid_ch': {'values': [32]}, 
					'stt_ch': {'values': [8, 16, 32, 64, 128]},
					'tilda_alpha': {'values': [200, 400, 600, 800, 1000, 1e10]}, 
				}, 
			}
			sweep_id = wandb.sweep(sweep_config, project=args.exp_name)
		wandb.agent(sweep_id, function=sweep, count=1)
	

					