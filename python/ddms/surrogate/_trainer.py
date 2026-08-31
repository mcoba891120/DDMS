import numpy as np
import torch
import logging
from torch_geometric.loader import DataLoader
from typing import *
from argparse import Namespace
from .processing import Scaler
from ._tracker import Tracker

class Trainer():
	"""
	Do typical machine learning training/testing
	"""
	def __init__(self, cfg: Namespace):
		"""
		Initializes the Trainer class with the provided configuration.

		Args:
			cfg: A namespace containing configuration parameters.
		"""
		self.cfg = cfg
		self.prm = Namespace()
		formatter = logging.Formatter('%(levelname)s(%(name)s): %(message)s')
		handlerS = logging.StreamHandler()
		handlerS.setFormatter(formatter)
		level = logging.DEBUG if cfg.verbose==1 else logging.WARNING
		self.logger = logging.Logger(name='trainer', level=level)
		self.logger.addHandler(handlerS)
		self.logger.info(f'{"mode":<20s}: {self.cfg.mode}')
		self.logger.info(f'{"exp_name":<20s}: {self.cfg.exp_name}')
		self.logger.info(f'{"run_name":<20s}: {self.cfg.run_name}')
		self.logger.info(f'{"run_id":<20s}: {self.cfg.run_id}')
		self.logger.info(f'{"save_root":<20s}: {self.cfg.save_root}')
		self.logger.info(f'{"data_path":<20s}: {self.cfg.data_path}')
		self.logger.info(f'{"specify_data":<20s}: {self.cfg.specify_data}')
		self.logger.info(f'{"backend":<20s}: {self.cfg.backend}')
		self.logger.info(f'{"epochs":<20s}: {self.cfg.epochs}')
		self.logger.info(f'{"batch_size":<20s}: {self.cfg.batch_size}')
		self.logger.info(f'{"learning_rate":<20s}: {self.cfg.learning_rate}')
		self.logger.info(f'{"device":<20s}: {self.cfg.device}')
	
	# >---------------------------------------------------------------------------------------------------
	def _check_mode(modes:Union[str, List[str]], reverse:bool=False):
		"""
		A decorator function to conditionally execute a method based on the current mode.

		Parameters:
			modes: A list of modes or a single mode as a string.
			reverse: If True, the method will be executed when the current mode is not in 'modes'.
					 If False (default), the method will be executed only when the current mode is in 'modes'.

		Returns:
			decorator: The decorator function that can be used to wrap a method.

		Example:
			```
			@_check_mode(['mode1', 'mode2'], reverse=False)
			def my_method(self, *args, **kwargs):
				# Your method implementation here
				pass
			```
		"""
		if isinstance(modes, str):
			modes = [modes]
		def decorator(func):
			def wrapper(self, *args, **kwargs):
				skip = self.cfg.mode in modes if reverse else self.cfg.mode not in modes
				if skip:
					self.logger.info(f'{func.__name__:<20s}: passed in {self.cfg.mode} mode !')
					return 
				func(self, *args, **kwargs)
			return wrapper
		return decorator

	# >---------------------------------------------------------------------------------------------------
	def init(self,
	  		 Model,
			 Dataset,
	  		 hyperparams:Dict=None,
			 train_split:Union[float, int]=0.7,
			 shuffle:bool=True,
			 split_fn:Callable=None):
		"""
        Initialize an instance of MyClass with specified settings.

        Args:
            Model: The model class to be used.
            Dataset: The dataset class to be used for training and evaluation.
            hyperparams: Hyperparameters for the model, can be empty only in test mode. Default is None.
            train_split: The proportion of data to be used for training. Default is 0.7.
            shuffle: Whether to shuffle the dataset before splitting. Default is True.
            split_fn: Optional `(dataset, train_split, seed) -> (train_idx, val_idx)`
                callable overriding the default random shuffle+slice split,
                e.g. `ddms.surrogate.validation.stratified_split`. Leave as
                None to keep the existing random-split behavior.
        """
		self.set_tracker()
		self.set_dataset(Dataset, train_split, shuffle, split_fn=split_fn)
		self.set_model(Model, hyperparams)
		self.set_loss()
		self.set_optimizer()

	@_check_mode(['train', 'sweep', 'debug'])
	def fit(self, 
	 		report_every_nstep:int=5, 
			validate_every_nepoch:int=5, 
			plot_every_nepoch:int=1000, 
			viz_callbacks:List[Callable]=None):
		"""
		Trains the model.

		Args:
			report_every_nstep: The number of steps between reporting metrics.
			validate_every_nepoch: The number of epochs between validating the model.
			plot_every_nepoch: The number of epochs between plotting the model's performance.
			viz_callbacks: A list of callbacks that will be called when plotting the model's performance.
		"""
		for epoch in self.epoch_updater():
			for step in self.data_updater('train'):
				self.model.train()
				self.step(backward=True)
				if step%report_every_nstep==0 or step==self.prm.train_nsteps-1:
					self.report_metrics(step=step, tag='train')
				self.log_metrics(tag='train')

			if epoch%validate_every_nepoch==0 or epoch==self.cfg.epochs-1:
				for step in self.data_updater('val'):
					self.model.eval()
					self.step(backward=False)
					if step%report_every_nstep==0 or step==self.prm.val_nsteps-1:
						self.report_metrics(step=step, tag='val')
				self.log_metrics(tag='val')

				if self.is_improved():
					self.log_model()
				if epoch%plot_every_nepoch==0:
					self.log_figure(tag='val', viz_callbacks=viz_callbacks)
				
				for scheduler in self.schedulers.values():
					scheduler.step(self.avg_loss)		# TODO

	@_check_mode('test')
	def evaluate(self, 
	      		 report_every_nstep:int=2, 
				 viz_callbacks:List[Callable]=None):
		"""
		Evaluates the model on the test set.

		Args:
			report_every_nstep: The number of steps between reporting metrics.
			viz_callbacks: A list of callbacks that will be called when plotting the model's performance.
		"""
		with torch.no_grad():
			next(self.epoch_updater())
			for step in self.data_updater('test'):
				self.model.eval()
				self.step(backward=False)
				if step%report_every_nstep==0 or step==self.prm.val_nsteps-1:
					self.report_metrics(step=step, tag='test')
			self.log_metrics(tag='test')
			self.log_figure(tag='test', viz_callbacks=viz_callbacks)

	# >---------------------------------------------------------------------------------------------------	
	def set_tracker(self):
		"""Sets the tracker and load checkpoint if `run_id` is given."""
		self.logger.info(f'{"set_tracker":<20s}: {self.cfg.backend}')

		if self.cfg.mode!='debug' or (self.cfg.mode=='debug' and self.cfg.run_id):
			self.tracker = Tracker(backend=self.cfg.backend)
			self.tracker.start_run(self.cfg)

		if self.cfg.run_id:
			self.state_dict = self.tracker.load_state_dict('checkpoint', map_location=self.cfg.device)
			self.config_dict = self.tracker.load_dict('checkpoint/configs.json')

	def set_dataset(self,
		 			Dataset,
		 			train_split:Union[float, int]=0.7,
					shuffle:bool=True,
					seed:int=0,
					split_fn:Callable=None):
		"""
		Sets the dataset. If `specify_data` is given, only those `data.desc` contains `specify_data` will remain.

		Args:
			Dataset: The dataset class.
			train_split: The train/val split ratio, should be set as 0 while testing.
			shuffle: Whether to shuffle the dataset.
			seed: The random seed.
			split_fn: Optional `(dataset, train_split, seed) -> (train_idx, val_idx)`
				callable overriding the default random shuffle+slice split
				below, e.g. `ddms.surrogate.validation.stratified_split`.
		"""
		self.logger.info(f'{"set_dataset":<20s}: {self.cfg.specify_data if self.cfg.specify_data[0] else "all"}')

		# init dataset
		dataset = Dataset(self.cfg.data_path)

		# data normalization
		if self.cfg.run_id:
			self.scaler = Scaler().load_state_dict(self.state_dict['scaler'])
		else:
			self.scaler = dataset.scaler
		dataset.data = self.scaler.transform(dataset.data)
		self.scaler = self.scaler.to(self.cfg.device)

		# split
		torch.manual_seed(seed)
		if split_fn is not None:
			train_idx, val_idx = split_fn(dataset, train_split, seed)
			if len(train_idx)>0 and len(val_idx)>0:
				train_dataset = dataset[train_idx].copy()
				val_dataset = dataset[val_idx].copy()
			else:
				self.logger.info(f'{"set_dataset":<20s}: train/val dataset are set as the same !')
				train_dataset = dataset.copy()
				val_dataset = dataset.copy()
		else:
			if shuffle:
				dataset, perm = dataset.shuffle(return_perm=True)
				dataset = dataset[perm].copy()
			spl_idx = int(train_split*len(dataset))
			if len(dataset)>1 and (spl_idx>0 and spl_idx<=len(dataset)):
				train_dataset = dataset[:spl_idx].copy()
				val_dataset = dataset[spl_idx:].copy()
			else:
				self.logger.info(f'{"set_dataset":<20s}: train/val dataset are set as the same !')
				train_dataset = dataset.copy()
				val_dataset = dataset.copy()

		# specify data
		if self.cfg.specify_data:
			def _get_specified_dataset(dataset, specify_data):
				matched_idx = [i for i, d in enumerate(dataset.data.desc) for s in specify_data if s in d]
				if len(matched_idx)==0:
					self.logger.warning('specifying data but none of them matched !')
					return dataset
				return dataset[matched_idx].copy()

			train_dataset = _get_specified_dataset(train_dataset, self.cfg.specify_data)	
			val_dataset = _get_specified_dataset(val_dataset, self.cfg.specify_data)	

		kwargs = {'batch_size': self.cfg.batch_size, 'pin_memory': True}
		self.train_dataloader = DataLoader(train_dataset, shuffle=True, **kwargs)
		self.val_dataloader = DataLoader(val_dataset, shuffle=False, **kwargs)
		self.prm.train_nsteps = len(self.train_dataloader)
		self.prm.val_nsteps = len(self.val_dataloader) 

	def set_model(self, Model, hyperparams:Dict=None):
		"""
		Sets the model and load model checkpoint if `run_id` is given.

		Args:
			Model: The model class.
			hyperparams: The model hyperparameters.
		"""
		if self.cfg.run_id:
			self.prm.hyperparams = self.config_dict['parameter']['hyperparams']
			self.model = Model(**self.prm.hyperparams).to(self.cfg.device)
			self.model.load_state_dict(self.state_dict['model'])
		else:
			if hyperparams is None:
				self.logger.error('model hyperparams are not given !')
				raise ValueError
			self.prm.hyperparams = hyperparams
			self.model = Model(**self.prm.hyperparams).to(self.cfg.device)
		self._model_summary() 

		self.log_params(self.prm.hyperparams)

	def set_loss(self):
		"""Sets the loss function according to `configure_loss()` defined in model."""
		config = self.model.configure_loss()
		desc = config.get('desc')
		self.logger.info(f'{"loss":<20s}: {desc}')

		self.criterion = config.get('criterion')

	def set_optimizer(self):
		"""Sets the optimizer/scheduler according to `configure_optimizer()` defined in model."""
		config = self.model.configure_optimizer(lr=self.cfg.learning_rate, num_epochs=self.cfg.epochs, num_batches_per_epoch=len(self.train_dataloader))
		desc = config.get('desc')
		self.logger.info(f'{"optimizer":<20s}: {desc}')

		self.optimizers = config.get('optimizers')
		self.schedulers = config.get('schedulers')

	# >---------------------------------------------------------------------------------------------------
	def epoch_updater(self):
		"""Creates an iterator over the epochs."""
		epoch_start = 0
		if self.cfg.run_id:
			epoch_start = self.config_dict['parameter']['epoch']

		for epoch in range(epoch_start, self.cfg.epochs):
			self.prm.epoch = epoch
			yield epoch

	def data_updater(self, mode:str, reset_tracking:bool=True):
		"""
		Creates an iterator over the dataloader for the specified mode.

		Args:
			mode: The mode, either `train`, `val`, or `test`.
			reset_tracking: Whether to reset the tracking history.
		"""
		if reset_tracking:
			self._reset_tracking()
		if mode=='train':
			return self._data_updater(self.train_dataloader)
		if mode=='val' or mode=='test':
			return self._data_updater(self.val_dataloader)
		
	def step(self, backward:bool=False):
		"""
		Performs a step of the training or evaluation.

		Args:
			backward: Whether to perform backpropagation.
		"""
		kwloss = self.model.step_forward(self.data, self.criterion)
		if backward:
			self.model.step_backward(kwloss, self.optimizers)
		self._register_tracking(kwloss)

	# >---------------------------------------------------------------------------------------------------
	def report_metrics(self, step:int, tag:str='train'):
		"""
		Reports the metrics for the specified tag.

		Args:
			step: The current step.
			tag: The tag, either `train` or `val` or `test`.
		"""
		niters = self.prm.train_nsteps if 'train' in tag else self.prm.val_nsteps
		msg1 = f'tag: {tag} | epoch: [{self.prm.epoch+1}/{self.cfg.epochs}] | step: {step+1}/{niters}'
		msg2 = f'loss: {self.avg_loss:.4f} | mape: {self.avg_mape:.2%}'
		self.logger.info(f'{msg1} | {msg2}')

	@_check_mode('debug', reverse=True)
	def log_metrics(self, tag:str='train'):
		"""
		Logs the metrics for the specified tag.

		Args:
			tag: The tag, either `train` or `val` or `test`.
		"""
		for k, v in self.trackings.items():
			self.tracker.log_metric(f'{tag}/{k}', np.mean(v), self.prm.epoch)

	@_check_mode('debug', reverse=True)
	def log_params(self, prms:Dict):
		"""
		Logs the parameters.

		Args:
			prms: The parameters to be logged.
		"""
		self.tracker.log_params(prms)

	@_check_mode('debug', reverse=True)
	def log_figure(self, tag:str='train', viz_callbacks:List[Callable]=None):
		"""
		Logs the figures.

		Args:
			tag: The tag, either `train` or `val` or `test`.
			viz_callbacks: The list of visualization callbacks.
		"""
		for callback in viz_callbacks:
			fig = callback(self.model, self.data, self.scaler)
			self.tracker.log_figure(fig, f'{tag}/{callback.name}', self.prm.epoch)
		
	@_check_mode('debug', reverse=True)
	def log_model(self):
		"""Logs the model."""
		state_dict = {
			'model': self.model.state_dict(), 
			'scaler': self.scaler.state_dict()
		}
		config_dict = {
			'config': vars(self.cfg),
			'parameter': vars(self.prm)
		}
		self.tracker.log_state_dict(state_dict, 'checkpoint')
		self.tracker.log_dict(config_dict, 'checkpoint/config.json')

	def is_improved(self):
		"""Checks if the model has improved."""
		if not hasattr(self.prm, 'best_loss') or self.avg_loss<self.prm.best_loss:
			self.prm.best_loss = self.avg_loss
			return True 
		return False

	def is_converged(self, patience:int=4):
		"""
		Checks if the model has converged.
		
		WARNING: 
			not been tested

		Args:
			patience: The patience, the number of epochs to wait before declaring convergence.
  		"""	
		if not hasattr(self, '_count'):
			self._count = 0

		if self.avg_loss>=self.prm.best_loss:
			self._count += 1

		if self._count>=patience:
			self._count = 0
			del self.prm.best_loss
			return True
			
		return False

	# >---------------------------------------------------------------------------------------------------
	def _model_summary(self):
		"""Prints a summary of the model."""
		params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
		self.logger.info(f'{"parameters":<20s}: {params}')
		self.logger.info(f'{"model":<20s}: {self.model}')
		self.log_params({'parameters': params})

	def _data_updater(self, dataloader):
		"""
		Creates an iterator over the dataloader.

		Args:
			dataloader: The dataloader.
		"""
		for step, data in enumerate(dataloader):
			self.data = data.to(self.cfg.device)
			yield step

	def _register_tracking(self, trackings:Dict):
		"""
		Registers the tracking metrics of model.

		Args:
			trackings: The trackings.
		"""
		if not hasattr(self, 'trackings'):
			self.trackings = {k: [v.item()] for k, v in trackings.items()}
			return 

		for k, v in trackings.items():
			self.trackings[k] += [v.item()]

	def _reset_tracking(self):
		"""Resets the tracking metrics of model."""
		if hasattr(self, 'trackings'):
			for k in self.trackings:
				self.trackings[k] = []

	# >---------------------------------------------------------------------------------------------------
	@property
	def avg_loss(self):
		"""Returns the average loss."""
		return np.mean(self.trackings.get('loss'))
	@property
	def avg_mape(self):
		"""Returns the average MAPE"""
		return np.mean(self.trackings.get('mape'))
	@property
	def avg_acc(self):
		"""Returns the average accuracy."""
		return np.mean(self.trackings.get('accuracy'))
	

# region ToBeUpdate
# >---------------------------------------------------------------------------------------------------
# > TODO: following code requires update
# >---------------------------------------------------------------------------------------------------
	@property
	def avg_corrcoef(self):
		""" 
		WARNING: deprecated 
			pearson correlation coefficient 
		"""
		Rs = []
		for y_pred, y in zip(self.y_pred.cpu().numpy(), self.y.cpu().numpy()):
			r = np.corrcoef(y_pred, y, rowvar=False)
			if not np.any(np.isnan(r)):
				Rs.append(r)
		return np.array(Rs).mean(axis=0) 
	@property
	def avg_k(self):
		""" 
		WARNING: deprecated
			average DEQ forward iteration 
		"""
		return np.mean(self.trackings.get('Records/DEQiter').get('iteration'))
	@property
	def avg_res(self):
		""" 
		WARNING: deprecated
			average DEQ forward residual 
		"""
		res = self.trackings.get('Records/DEQres').get('residual')
		m_res = np.ma.masked_array(res, np.isnan(res), fill_value=np.inf)
		return np.mean(m_res.filled())
# endregion ToBeUpdate
