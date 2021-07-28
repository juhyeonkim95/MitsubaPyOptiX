from core.emitters.emitter import Emitter
from core.loader.loader_general import *


class SpotEmitter(Emitter):
	def __init__(self, props):
		super().__init__(props)
		self.intensity = load_value(props, "intensity", [1, 1, 1])
		self.cutoff_angle = load_value(props, "cutoffAngle", 20)
		self.beam_width = load_value(props, "beamWidth", 0.75 * self.cutoff_angle)

	def __str__(self):
		logs = [
			"[Emitter]",
			"\t- type : %s" % "spot",
			"\t- intensity : %s" % str(self.intensity),
			"\t- cutoff_angle : %s" % str(self.cutoff_angle),
			"\t- beam_width : %s" % str(self.beam_width),
		]
		return "\n".join(logs)

	def __array__(self):
		np_array = np.zeros(1, dtype=Emitter.dtype)
		np_array["lightType"] = 4
		# TODO : fill in
		#np_array["position"] = np.array(self.position, dtype=np.float32)
		#np_array["intensity"] = np.array(self.intensity, dtype=np.float32)
		return np_array
