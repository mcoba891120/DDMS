import abc
from typing import List
from torch.nn import Module

class BaseModule(Module):
	def __init__(self):
		super().__init__()

	@abc.abstractmethod
	def configure_loss(self) -> dict:
		""" 
		configure loss criterion for `step_forward`

		Return {
			'criterion': {'loss': L1Loss(), 'mape': Mape()}
			'desc': description of loss configuration, e.g., 'MAE'
		}
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def configure_optimizer(self, lr) -> dict:
		"""
		configure optimizer and scheduler for `step_backward`

		Return {
			'optimizers': {'optimizer1': Adam(), 'optimizer2': RMSprop()}
			'schedulers': {'scheduler1': ReduceLROnPlateau()}
			'desc': description of optimizer configuration, e.g., 'Adam'
		}
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def step_forward(self, data, criterion:dict, **shared_kwargs) -> dict:
		"""
		forward step util get loss, in which
		- criterion is defined in `configure_loss`
		- shared_kwargs is the correspond keyword arguments in `_register_sharing_args`

		Return {
			'loss1': ...,
			'mape': ...
		}
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def step_backward(self, kwloss:dict, optimizers:dict, **shared_kwargs) -> None:
		""" 
		backward step to update model parameters, in which
		- kwloss is given from `step_forward`
		- optimizers is defined in `configure_optimizer`
		- shared_kwargs is the correspond keyword arguments in `_register_sharing_args`
		"""
		raise NotImplementedError
