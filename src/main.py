import sys
from utils.config_utils import *
from core.renderer_constants import process_config


if __name__ == "__main__":
	argument = sys.argv
	if len(argument) > 1:
		config_file = argument[1]
	else:
		config_file = "../configs_example/brdf.json"
	config = load_config_recursive(config_file)

	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	from pyoptix import Compiler
	Compiler.clean()
	Compiler.keep_device_function = False
	file_dir = os.path.dirname(os.path.abspath(__file__))
	Compiler.add_program_directory(file_dir)

	from core.renderer import Renderer
	renderer = Renderer()
	renderer.render(**config)
