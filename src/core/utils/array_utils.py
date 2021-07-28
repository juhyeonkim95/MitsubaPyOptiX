import numpy as np

from core.textures.texture import Texture


def insert_texture_or_spectrum(array, value, spectrum_key, texture_key=None):
	if texture_key is None:
		texture_key = spectrum_key + "_texture_id"
	if isinstance(value, Texture):
		array[texture_key] = value.list_index
	elif isinstance(value, float) or len(value) == 1:
		array[texture_key] = -1
		array[spectrum_key] = np.repeat(value, 3).astype(np.float32)
	elif len(value) == 3:
		array[texture_key] = -1
		array[spectrum_key] = np.array(value, dtype=np.float32)


def insert_texture_or_spectrum_single_value(array, value, spectrum_key, texture_key=None):
	if texture_key is None:
		texture_key = spectrum_key + "_texture_id"
	if isinstance(value, Texture):
		array[texture_key] = value.list_index
	elif isinstance(value, float):
		array[texture_key] = -1
		array[spectrum_key] = value
	else:
		np_array = np.array(value, dtype=np.float32)
		np_array_mean = np.mean(np_array)
		array[texture_key] = -1
		array[spectrum_key] = np_array_mean
