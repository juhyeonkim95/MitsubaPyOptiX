from enum import IntEnum


class BSDFFlags(IntEnum):
	diffuse = 1
	dielectric = 1 << 1
	rough_dielectric = 1 << 2
	conductor = 1 << 3
	rough_conductor = 1 << 4
	plastic = 1 << 5
	rough_plastic = 1 << 6