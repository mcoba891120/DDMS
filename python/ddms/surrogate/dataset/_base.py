import torch
import os
import abc 
from tqdm import tqdm
from torch_geometric.data import Data, Dataset, InMemoryDataset
from typing import Dict, List
from ..processing import Scaler

class MemoryDataset(InMemoryDataset):
	"""
	base class of In-memory dataset, load all data in cpu memory

	Args:
		root: root path where the dataset should be saved
		along: along which feature to calculate data scaler 
	Create:
		data: data containing attributes defined in `load_data`
		data_scale: Data({data.keys: [min, max]}), data min/max according to `along`
	"""

	# WARNING: bug while torch.save large tensor
	# torch version: 1.7.1+cu110
	# issue tarcking: https://github.com/pytorch/pytorch/issues/47201
	LARGE_DATA = True 

	def __init__(self, root:str):
		self._clearing(root)
		super().__init__(root)
		if self.LARGE_DATA:
			self.data, self.slices = self.data_, self.slices_
		else:
			self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

		self.scaler = Scaler(self._scale_along(self.data), self._scale_type)
		self.scaler.fit(self.data)

	def _clearing(self, root):
		""" clearing weird file that causes "slices bug" """
		for weird_file in ['pre_transform.pt', 'pre_filter.pt']:
			try: os.remove(f'{root}/processed/{weird_file}')
			except FileNotFoundError: pass
	
	@property
	@abc.abstractmethod
	def processed_file_names(self):
		raise NotImplementedError

	@abc.abstractmethod
	def _load_data(self) -> List[Data]:
		""" loading raw data list """
		raise NotImplementedError

	@abc.abstractmethod
	def _scale_type(self) -> str:
		""" `minmax` or `standard` or `lmsc` """
		raise NotImplementedError

	@abc.abstractmethod
	def _scale_along(self, data) -> Dict[str, List[slice]]:
		""" along which feature to calculate data scaler  """
		raise NotImplementedError

	def process(self):
		# get data
		data_list = self._load_data()
		data, self.slices = self.collate(data_list)

		if self.LARGE_DATA:
			self.data_, self.slices_ = data, self.slices
		else:
			torch.save((data, self.slices), self.processed_paths[0])


class IterateDataset(Dataset):
	"""
	#### WARNING: LEGACY

	base class of iterate dataset, load all data iteratively

	Args:
		root: root path where the dataset should be saved
		along: along which feature to calculate data scaler 
	Create:
		data: data containing attributes defined in `load_data`
		data_scale: Data({data.keys: [min, max]}), data min/max according to `along`
	"""
	def __init__(self, root:str):
		self._clearing(root)
		super().__init__(root, transform=self._transform_func)
		self.data_scale = torch.load(self.processed_paths[0])

	def _clearing(self, root):
		""" clearing weird file that causes "slices bug" """
		for weird_file in ['pre_transform.pt', 'pre_filter.pt']:
			try: os.remove(f'{root}/processed/{weird_file}')
			except FileNotFoundError: pass
	
	@property
	@abc.abstractmethod
	def raw_file_names(self):
		raise NotImplementedError

	@property
	@abc.abstractmethod
	def processed_file_names(self):
		raise NotImplementedError

	@abc.abstractmethod
	def len(self):
		raise NotImplementedError
	
	@abc.abstractmethod
	def get(self, idx):
		raise NotImplementedError

	@abc.abstractmethod
	def _load_data(self, idx) -> Data:
		""" loading raw data """
		raise NotImplementedError

	@abc.abstractmethod
	def _scale_along(self, data) -> Dict[str, List[slice]]:
		""" along which feature to calculate data scaler  """
		raise NotImplementedError

	def process(self):
		if os.path.exists(self.processed_paths[0]):
			data_scale = torch.load(self.processed_paths[0])
		else:
			data_scale = None
			for idx in tqdm(range(self.len())):
				data = self._load_data(idx)
				along = self._scale_along(data)
				data_scale = self._trace_min_max(data, along, data_scale)
			torch.save(data_scale, self.processed_paths[0])

	def _trace_min_max(self, data:Data, along:Dict[str, List[slice]], data_scale:Data=None):
		""" 
		trace min max of data, assuming last dimension of data represents "features"

		Args:
			data: new Data object to compare with history data_scale
			data_scale: history min/max of data
			along: data key and feature slices pair to calculate min/max
		"""
		if data_scale is None:
			data_scale = {}
			for k in along.keys():
				data_scale[k] = torch.tensor([[float('inf'), float('-inf')]]*data[k].size(-1)) 				# (feature_dim, 2)
		
		for k in along.keys():
			for s in along[k]:
				data_scale[k][s,0] = torch.minimum(data_scale[k][s,0].min(), data[k][...,s].min())	# (slice_feature_dim, )
				data_scale[k][s,1] = torch.maximum(data_scale[k][s,1].max(), data[k][...,s].max())	# (slice_feature_dim, )

		return data_scale

	def _transform_func(self, data:Data):
		""" normalize to [0,1] """
		along = self._scale_along(data)
		for k in along.keys():
			Min = self.data_scale[k][:,0] 																# (feature_dim)
			Max = self.data_scale[k][:,1] 																# (feature_dim)
			data[k] = data[k].sub(Min).div(Max-Min)
		return data

	

