# Usage
PyDDMS contains two modules, which are `numeric` & `surrogate`. `numeric` handles all of the tasks related to DNS while `surrogate` handles all of the tasks related to DDMS. Both of the module offers high-level utility class for tasks control, i.e., `Handler` & `Trainer` for `numeric` & `surrogate` respectively.

## Initialization
=== "numeric"
	``` py linenums="1" title="numeric/main.py"

	import sys 
	sys.path.append('../../python') 			# (1)
	import argparse 							# (2)
	from ddms.numeric import UtilInterface 		# (3)

	# handle input arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inputyaml', type=str, required=True, dest='yaml')
	parser.add_argument('-m', '--mode', type=str, default='test', dest='mode')
	parser.add_argument('-s', '--solver', type=str, default='damask', dest='solver')
	parser.add_argument('-tn', '--taskname', type=str, default='', dest='taskname')
	parser.add_argument('-t', '--target', type=str, default='', dest='target')
	parser.add_argument('-rd', '--ROOTdir', type=str, default='', dest='ROOTdir')
	args = parser.parse_args()

	mode 		= args.mode
	solver 		= args.solver
	taskname 	= args.taskname
	target 		= args.target
	ROOTdir  	= args.ROOTdir
	YAML 		= f'../../config/{args.yaml}.yaml'

	# initialize a numeric `UTL` instance with given arguments
	UTL = UtilInterface(YAML,
						taskname = taskname,
						target   = target,
						ROOTdir  = ROOTdir,
						solver   = solver)
	```

	1. add PyDDMS package to system path
	2. use "anything" to handle input arguments, e.g., `argparse` here
	3. `UtilInterface` handles all of the functionality with given arguments

	???+ info "Arguments"
		...

=== "surrogate" 
	``` py linenums="1" title="surrogate/main.py"
	import sys 
	sys.path.append('../../python')			# (1)
	import argparse							# (2)
	from ddms.surrogate import Trainer		# (3)

	# handle input arguments
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

	# initialize a surrogate `Trainer` instance with given arguments
	trainer = Trainer(args)
	```

	1. add PyDDMS package to system path
	2. use "anything" to handle input arguments, e.g., `argparse` here
	3. `Trainer` handles all of the functionality with given arguments

	???+ info "Arguments"
		`mode`
		:	switch between mode, train | test | debug | sweep
		
		`verbose`
		:	verbose level, 1 for debug, 0 for only warning or error
		
		`exp_name`
		:	name of mlflow experiment
		
		`run_name`
		:	name of mlflow run
		
		`run_id`
		:	mlflow run_id to resume and load pretrain model from
		
		`sweep_id`
		:	wandb sweep_id
		
		`save_root`
		:	root path for saving cache
		
		`data_path`
		:	dataset path
		
		`specify`
		:	data names to contained in dataset, can be multiple
		
		`sync`
		:	offline or online, only used with wandb backend
		
		`backend`
		:	choose either mlflow or wandb backend
		
		`epochs`
		:	epochs used in this run
		
		`batch_size`
		:	batch size used in this run
		
		`learning_rate`
		:	learning rate used in this run
		
		`device`
		:	cuda or cpu