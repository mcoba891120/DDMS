import torch 
import mlflow 
import wandb 
import json

class Tracker():
	""" Track training/testing information using `mlflow` or `wandb` backend """
	def __new__(self, backend='wandb'):
		"""
		Create a new Tracker object.

		Args:
			backend: The backend to use. Must be one of "mlflow" or "wandb".

		Returns:
			A new Tracker object.
		"""
		assert backend in ['mlflow', 'wandb'], 'unsupported backend! choose mlflow or wandb!'
		if backend=='mlflow':
			return _Tracker_mlflow()
		if backend=='wandb':
			return _Tracker_wandb()

class _Tracker_wandb():
	def __init__(self):
		if wandb.run is not None:
			wandb.finish()

	def start_run(self, cfg):
		"""
		Starts a new Wandb run.

		Args:
			cfg: The trainer configuration.
		"""
		wandb.init(
			project=cfg.exp_name,
			name=cfg.run_name,
			job_type=cfg.job_type if hasattr(cfg, 'job_type') else cfg.mode,
			config=vars(cfg),
			dir=f'{cfg.save_root}',
			mode=cfg.sync)
	
	def log_params(self, params):
		"""
		Logs the parameters of the model.

		Args:
			params: The parameters to be logged.
		"""
		wandb.config.update(params)

	def log_metric(self, k, v, step):
		"""
		Logs a metric.

		Args:
			k: The name of the metric.
			v: The value of the metric.
			step: The step at which the metric was recorded.
		"""
		wandb.log({k: v}, step=step)

	def log_figure(self, fig, art_file, step):
		"""
		Logs a figure.

		Args:
			fig: The figure to log.
			art_file: The name of the figure.
			step: The step at which the figure was recorded.
		"""

		wandb.log({art_file: fig}, step=step)

	def log_state_dict(self, state_dict, art_path):		
		"""
		Logs the state dict of the model.

		Args:
			state_dict: The state dict of the model.
			art_path: The name of the state dict, not used in `wandb` backend.
		"""
		art_name = f'{wandb.run.name}_model'
		art = wandb.Artifact(art_name, type='model')
		with art.new_file('state_dict.pth', 'wb') as f:
			torch.save(state_dict, f)
		wandb.log_artifact(art, aliases=[wandb.run.id])

	def log_dict(self, config_dict, art_file):
		"""
		Logs a dictionary.

		Args:
			config_dict: The dictionary to log.
			art_file: The name of the dictionary, not used in `wandb` backend.
		"""
		art_name = f'{wandb.run.name}_config'
		art = wandb.Artifact(art_name, type='config')
		with art.new_file('config.json', 'w') as f:
			json.dump(config_dict, f)
		wandb.log_artifact(art, aliases=[wandb.run.id])

	def load_state_dict(self, art_path, **kwargs):
		"""
		Loads the state dict of the model.

		Args:
			art_path: The name of the state dict, not used in `wandb` backend.
			**kwargs: Keyword arguments passed to `torch.load()`.

		Returns:
			The state dict of the model.
		"""
		art_name = f'{wandb.run.name}_model:{wandb.config.run_id}'
		art_dir = f'{wandb.run.dir}/artifacts'
		art_uri = wandb.use_artifact(art_name, type='model').download(art_dir)
		state_dict = torch.load(f'{art_uri}/state_dict.pth', **kwargs) 
		return state_dict
	
	def load_dict(self, art_file):
		"""
		Loads a dictionary.

		Args:
			art_file: The name of the dictionary, not used in `wandb` backend.

		Returns:
			The dictionary.
		"""
		art_name = f'{wandb.run.name}_config:{wandb.config.run_id}'
		art_dir = f'{wandb.run.dir}/artifacts'
		art_uri = wandb.use_artifact(art_name, type='config').download(art_dir)
		with open(f'{art_uri}/config.json', 'r') as f:
			config_dict = json.load(f) 
		return config_dict
	
	def get_artifact_uri(self, art_path=None):
		"""
		Gets the artifact URI for the given path.

		Args:
			art_path: The sub-path of the artifact within `artifacts` folder.

		Returns:
			The artifact URI.
		"""		
		if art_path is None:
			return f'{wandb.run.dir}/artifacts'
		return f'{wandb.run.dir}/artifacts/{art_path}'


class _Tracker_mlflow():
	"""
	WARNING: Legacy code
		The functionality has not been tested, consider using wandb instead.
	"""
	def __init__(self):
		if mlflow.active_run():
			mlflow.end_run()

	def start_run(self, cfg):
		mlflow.set_tracking_uri(f'{cfg.save_root}/mlruns')
		exp_id = mlflow.set_experiment(cfg.exp_name).experiment_id
		mlflow.start_run(
			experiment_id=exp_id, 
			run_id=cfg.run_id,
			run_name=cfg.run_name,
			tags=vars(cfg))
	
	def log_params(self, params):
		mlflow.log_params(params)

	def log_metric(self, k, v, step):
		mlflow.log_metric(k, v, step)

	def log_figure(self, fig, art_file, step):
		mlflow.log_figure(fig, f'{art_file}_epoch{step}.png')

	def log_state_dict(self, state_dict, art_path):
		mlflow.pytorch.log_state_dict(state_dict, art_path)

	def log_dict(self, config_dict, art_file):
		mlflow.log_dict(config_dict, art_file)

	def load_state_dict(self, art_path, **kwargs):
		state_dict_uri = mlflow.get_artifact_uri(art_path)
		return mlflow.pytorch.load_state_dict(state_dict_uri, **kwargs)
	
	def load_dict(self, art_file):
		config_dict_uri = mlflow.get_artifact_uri(art_file)
		return mlflow.artifacts.load_dict(config_dict_uri)
	
	def get_artifact_uri(self, art_path=None):
		return mlflow.get_artifact_uri(artifact_path=art_path)

