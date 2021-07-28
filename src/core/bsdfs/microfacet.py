from enum import IntEnum


class DistributionType(IntEnum):
	"""
	Microfacet normal distribution. Three types are supported
	1) beckmann
	2) ggx
	3) phong
	"""
	beckmann = 0
	ggx = 1
	phong = 2


