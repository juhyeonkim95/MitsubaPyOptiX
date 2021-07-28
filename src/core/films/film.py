class Film:
	def __init__(self, props):
		from core.loader.loader_general import load_value
		self.type = props.attrib["type"]
		self.width = load_value(props, "width", 768)
		self.height = load_value(props, "height", 576)
		self.file_format = load_value(props, "fileFormat", "png")
		self.pixel_format = load_value(props, "pixelFormat", "rgb")
		self.tonemap_method = load_value(props, "tonemapMethod", "gamma")
		self.gamma = load_value(props, "gamma", -1)
		self.exposure = load_value(props, "exposure", 0)
		self.key = load_value(props, "key", 0.18)
