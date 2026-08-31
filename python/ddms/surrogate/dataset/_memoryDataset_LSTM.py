import os
import sys
from tqdm import tqdm
from multiprocessing import Pool
from ._base import MemoryDataset
from ..processing import get_dev5_data

class MemoryDataset_LSTM(MemoryDataset):
	"""
	Args:
		root: root path where the dataset should be saved
	Create:
		data: Data
			
		data_minmax: Data({data.keys: [min, max]})
	"""
	LARGE_DATA = False

	def __init__(self, root):
		super().__init__(root)

	@property
	def result_dir(self):
		return os.path.join(self.root, 'result')

	@property
	def processed_file_names(self):
		return ['dev5.pt']
	
	@property
	def _scale_type(self) -> str:
		return 'lmsc'

	def _scale_along(self, data):
		""" along which feature to calculate data minmax  """
		return {'CSdev5': [slice(0, data.CSdev5.size(-1))]}

	def _get_data(self, folder):
		folder_path = os.path.join(self.raw_dir, folder)
		result_path = os.path.join(self.result_dir, folder)
		# return get_UCS_data(folder_path, result_path)
		return get_dev5_data(result_path, check_len=1010)

	def _load_data(self):
		data_list = []
		# folders = next(os.walk(self.raw_dir))[1]
		folders = next(os.walk(self.result_dir))[1]
		
		# multi-core
		with Pool(processes=1) as pool:		
			pool_iter = pool.imap_unordered(self._get_data, folders)
			for data in tqdm(pool_iter, desc='loading data', total=len(folders), file=sys.stdout):
				if data:
					data_list.append(data)

		return data_list

	# def _load_data(self):
	# 	import torch
	# 	from ..processing import reconstruct_data_list


	# 	stiff_data, stiff_slice = torch.load(os.path.join(stiff_root, 'dev5.pt'))
	# 	smooth_data, smooth_slice = torch.load(os.path.join(smooth_root, 'dev5.pt'))

	# 	# for i, desc in enumerate(smooth_data.desc):
	# 	# 	smooth_data.desc[i] = desc.split('\\')[-1] + '_smooth'

	# 	data_list = reconstruct_data_list(stiff_data, stiff_slice) + \
	# 				reconstruct_data_list(smooth_data, smooth_slice)
	# 	return data_list

	def process(self):
		""" required this interface to do process """
		super().process()

	
	
	

