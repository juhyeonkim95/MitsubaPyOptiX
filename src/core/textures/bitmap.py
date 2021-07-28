from core.textures.texture import Texture
import numpy as np
from enum import Enum


class BitmapTexture(Texture):
	def __init__(self, props):
		super().__init__(props)
		from core.loader.loader_general import load_value

		self.filename = load_value(props, "filename", default=None)
		self.uoffset = load_value(props, "uoffset", default=0)
		self.voffset = load_value(props, "voffset", default=0)
		self.uscale = load_value(props, "uscale", default=1)
		self.vscale = load_value(props, "vscale", default=1)

		self.gamma = load_value(props, "gamma", default=-1)
		self.texture_optix_id = -1

	def __array__(self):
		np_array = np.zeros(1, dtype=Texture.dtype)
		np_array['type'] = 1
		np_array['id'] = self.texture_optix_id
		uv_transform = np.array([
			[self.uscale, 0, self.uoffset],
			[0, self.vscale, self.voffset],
			[0, 0, 1]], dtype=np.float32)
		np_array['uv_transform'] = uv_transform
		np_array['srgb'] = 1
		return np_array

	def __str__(self):
		logs = [
			"[Texture]",
			"- type : bitmap",
			"\t- filename : %s" % self.filename,
			"\t- offset : (%.3f, %.3f)" % (self.uoffset, self.voffset),
			"\t- scale : (%.3f, %.3f)" % (self.uscale, self.vscale)
		]
		return "\n".join(logs)
