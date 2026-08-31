from ._base import BaseVizCallback
from ..tensor import dev5_to_m6
from ..visualize import VIZ, plot_homogenized

class VizHomogenizedCS6(BaseVizCallback):
	def __init__(self, name=None):
		super().__init__(name)
		
	def forward(self, model, data, scaler):
		batch 	= slice(0,1)
		y_pred	= model(data.dHSdev5)[0][batch]
		y 		= data.CSdev5[batch]
		CShyd6 	= data.CShyd6[batch]

		y_pred 	= scaler.inverse_transform(y_pred, 'CSdev5')
		y 		= scaler.inverse_transform(y, 'CSdev5')

		y_pred 	= dev5_to_m6(y_pred, CShyd6)/1e6
		y 		= dev5_to_m6(y, CShyd6)/1e6

		return plot_homogenized(y, y_pred, VIZ.names_IMP6, order='F')
		
class VizHomogenizedCSdev6(BaseVizCallback):
	def __init__(self, name=None):
		super().__init__(name) 

	def forward(self, model, data, scaler):
		batch 	= slice(0,1)
		y_pred 	= model(data.dHSdev5)[0][batch]
		y 		= data.CSdev5[batch]

		y_pred 	= scaler.inverse_transform(y_pred, 'CSdev5')
		y 		= scaler.inverse_transform(y, 'CSdev5')

		y_pred 	= dev5_to_m6(y_pred, 0)/1e6
		y 		= dev5_to_m6(y, 0)/1e6

		return plot_homogenized(y, y_pred, VIZ.names_IMP6, order='F')
