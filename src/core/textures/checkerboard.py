from core.textures.texture import Texture
import numpy as np


class CheckerBoard(Texture):
	def __init__(self, props):
		super().__init__(props)
		from core.loader.loader_general import load_value

		self.color0 = load_value(props, "color0", default=np.array([0.4], dtype=np.float32))
		self.color1 = load_value(props, "color1", default=np.array([0.2], dtype=np.float32))

		self.uoffset = load_value(props, "uoffset", default=0)
		self.voffset = load_value(props, "voffset", default=0)
		self.uscale = load_value(props, "uscale", default=1)
		self.vscale = load_value(props, "vscale", default=1)

	def __array__(self):
		np_array = np.zeros(1, dtype=Texture.dtype)
		np_array['type'] = 2
		uv_transform = np.array([
			[self.uscale, 0, self.uoffset],
			[0, self.vscale, self.voffset],
			[0, 0, 1]], dtype=np.float32)
		np_array['uv_transform'] = uv_transform
		np_array['color0'] = self.color0
		np_array['color1'] = self.color1
		return np_array

	def __str__(self):
		logs = [
			"[Texture]",
			"- type : checkerboard",
			"\t- color0 : %s" % self.color0,
			"\t- color1 : %s" % self.color1,
			"\t- offset : (%.3f, %.3f)" % (self.uoffset, self.voffset),
			"\t- scale : (%.3f, %.3f)" % (self.uscale, self.vscale)
		]
		return "\n".join(logs)
