import os
import sys
import torch
from tqdm import tqdm
from multiprocessing import Pool
from torch_geometric.data import Data
from ._base import MemoryDataset

class MemoryDataset_GNN(MemoryDataset):
	"""
	Args:
		root: root path where the dataset should be saved
	Expects, per RVE, `{root}/raw/{name}/data.pt` (Data with `euler`,
		`volume`, `num_neighbor`, `edge_index`, `edge_attr`, see
		`HandlerDamask.createGraphDataset`) and `{root}/raw/{name}/label.pt`
		(Tensor of shape (out_ch,)). Folders missing either file are skipped.
	Create:
		data: Data(x, edge_index, edge_attr, volume, y, batch, desc)
	"""
	LARGE_DATA = False

	def __init__(self, root):
		super().__init__(root)

	@property
	def processed_file_names(self):
		return ['data.pt']

	@property
	def _scale_type(self) -> str:
		return 'standard'

	def _scale_along(self, data):
		return {
			'x': [slice(0, data.x.size(-1))],
			'y': [slice(0, data.y.size(-1))],
		}

	def _get_data(self, folder):
		data_path = os.path.join(self.raw_dir, folder, 'data.pt')
		label_path = os.path.join(self.raw_dir, folder, 'label.pt')
		if not (os.path.exists(data_path) and os.path.exists(label_path)):
			return None

		graph = torch.load(data_path, weights_only=False)
		label = torch.load(label_path, weights_only=False)

		x = torch.cat([graph.euler[0], graph.volume, graph.num_neighbor], dim=-1)	# (num_nodes, 5)
		return Data(
			x=x.float(),
			edge_index=graph.edge_index,
			edge_attr=graph.edge_attr.float(),
			volume=graph.volume.float(),
			y=label.float().unsqueeze(0),		# (1, out_ch)
			desc=folder,
		)

	def _load_data(self):
		folders = next(os.walk(self.raw_dir))[1]

		data_list = []
		with Pool(processes=1) as pool:
			pool_iter = pool.imap_unordered(self._get_data, folders)
			for data in tqdm(pool_iter, desc='loading data', total=len(folders), file=sys.stdout):
				if data is not None:
					data_list.append(data)

		return data_list
