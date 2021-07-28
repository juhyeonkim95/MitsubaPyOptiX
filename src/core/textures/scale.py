from core.textures.texture import Texture
import numpy as np


class ScalingTexture(Texture):
	def __init__(self, props):
		super().__init__(props)
		from core.loader.loader_general import load_value, load_texture
		self.scale = load_value(props, "scale", default=1)
		self.texture = load_texture(props.find('texture'))

	def __array__(self):
		np_array = np.array(self.texture)
		return np_array

	def __str__(self):
		logs = [
			"[Texture]",
			"\t- type : scale",
			"\t- scale : %s" % str(self.scale),
			"\t- texture : %s" % str(self.texture)
		]
		return "\n".join(logs)
