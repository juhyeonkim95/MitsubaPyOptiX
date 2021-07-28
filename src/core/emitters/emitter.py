import numpy as np


class Emitter:
	# Static data type
	dtype = np.dtype([
		('position', np.float32, 3),
		('direction', np.float32, 3),
		('normal', np.float32, 3),
		('emission', np.float32, 3),
		('intensity', np.float32, 3),
		('u', np.float32, 3),
		('v', np.float32, 3),
		('radius', np.float32),
		('area', np.float32),
		('inv_area', np.float32),
		('cosTotalWidth', np.float32),
		('cosFalloffStart', np.float32),
		('lightType', np.uint32),
		('pos_buffer_id', np.int32),
		('indices_buffer_id', np.int32),
		('normal_buffer_id', np.int32),
		('n_triangles', np.int32),
		('transformation', np.float32, (4, 4)),
		('envmapID', np.int32),
		('isTwosided', np.int32)
	])

	def __init__(self, props):
		self.type = props.attrib["type"]
		self.list_index = -1

	def __array__(self):
		pass
