from core.emitters.emitter import Emitter
from core.loader.loader_general import *


class AreaLight(Emitter):
	def __init__(self, props):
		super().__init__(props)
		self.radiance = load_value(props, "radiance", [1, 1, 1])
		self.shape = None

	def __str__(self):
		logs = [
			"[Emitter]",
			"\t- type : %s" % "area",
			"\t- radiance : %s" % str(self.radiance)
		]
		return "\n".join(logs)

	def __array__(self):
		np_array = np.zeros(1, dtype=Emitter.dtype)
		self.shape.fill_area_light_array(np_array)
		np_array['inv_area'] = 1.0 / np_array['area']
		np_array["emission"] = np.array(self.radiance, dtype=np.float32)
		return np_array
