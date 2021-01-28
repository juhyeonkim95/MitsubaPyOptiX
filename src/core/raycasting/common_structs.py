import numpy as np


class RayData(object):
	dtype = np.dtype([
		('origin', np.float32, 3),
		('tmin', np.int32),
		('dir', np.float32, 3),
		('tmax', np.float32)
	])


class Hit(object):
	dtype = np.dtype([
		('t', np.float32),
		('geom_normal', np.float32, 3),
		('hit_point', np.float32, 3),
		('color', np.float32, 3),
		('attenuation', np.float32, 3),
		('new_direction', np.float32, 3),
		('pdf', np.float32),
		('done', np.uint32),
		('seed', np.uint32),
		('result', np.float32, 3)
	])

	"""
	__array__ is called when a BasicLight is being converted to a numpy array.
	Then, one can assign that numpy array to an optix variable/buffer. The format will be user format.
	Memory layout (dtype) must match with the corresponding C struct in the device code.
	"""
	# def __array__(self):
	# 	np_array = np.zeros(1, dtype=Hit.dtype)
	# 	np_array['t'] = self._t
	# 	np_array['triId'] = self._triID
	# 	np_array['u'] = self._u
	# 	np_array['v'] = self._v
	# 	np_array['geom_normal'] = self._geom_normal
	# 	# np_array['color'] = self._color
	# 	# np_array['casts_shadow'] = 1 if self._casts_shadow else 0
	# 	# np_array['padding'] = 0
	# 	return np_array
	#
	# def __init__(self, pos, color, casts_shadow):
	# 	self._pos = pos
	# 	self._color = color
	# 	self._casts_shadow = casts_shadow
