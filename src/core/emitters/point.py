from core.emitters.emitter import Emitter
from core.loader.loader_general import *


class PointLight(Emitter):
	def __init__(self, props):
		super().__init__(props)
		self.position = load_value(props, "position", None)
		if self.position is None:
			transform = load_value(props, "toWorld", None)
			self.position = transform * Vector3([0, 0, 0])
		self.intensity = load_value(props, "intensity", [1, 1, 1])

	def __str__(self):
		logs = [
			"[Emitter]",
			"\t- type : %s" % "point",
			"\t- position : %s" % str(self.position),
			"\t- intensity : %s" % str(self.intensity)
		]
		return "\n".join(logs)

	def __array__(self):
		np_array = np.zeros(1, dtype=Emitter.dtype)
		np_array["lightType"] = 2
		np_array["position"] = np.array(self.position, dtype=np.float32)
		np_array["intensity"] = np.array(self.intensity, dtype=np.float32)
		return np_array
