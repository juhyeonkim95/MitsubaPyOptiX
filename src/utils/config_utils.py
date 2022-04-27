import json
from pathlib import Path


def load_config_recursive(config_file_name):
	path = Path(config_file_name)
	parent_path = path.parent
	config = json.load(open(config_file_name))

	if "include" in config:
		config_include = load_config_recursive(parent_path / config["include"])
		config.pop("include")
		merged = {**config_include, **config}
		return merged
	else:
		return config