from core.emitters.emitter import Emitter
from core.loader.loader_general import *


class EnvironmentMap(Emitter):
	def __init__(self, props):
		super().__init__(props)
		self.filename = load_value(props, "filename", default=None)
		self.transform = load_value(props, "toWorld")
		self.envmapID = None

	def __str__(self):
		logs = [
			"[Emitter]",
			"\t- type : %s" % "envmap",
			"\t- filename : %s" % str(self.filename)
		]
		return "\n".join(logs)

	def __array__(self):
		np_array = np.zeros(1, dtype=Emitter.dtype)
		np_array["lightType"] = 5
		np_array["transformation"] = np.array(self.transform.transpose(), dtype=np.float32)
		np_array["envmapID"] = 1
		return np_array
