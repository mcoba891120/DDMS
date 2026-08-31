import abc
from torch.nn import Module
from torch_geometric.data import Data
from matplotlib.figure import Figure
from ..processing import Scaler

class BaseVizCallback(Module):
	def __init__(self, name=None):
		super().__init__() 
		self.name = name or self.__class__.__name__
	
	@abc.abstractmethod
	def forward(self, model:Module, data:Data, scaler:Scaler) -> Figure:
		"""
		base callback module for visualization

		Return matplotlib figure
		"""
		raise NotImplementedError


