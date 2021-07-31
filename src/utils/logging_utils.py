import logging
from datetime import timedelta


class NewLineFormatter(logging.Formatter):
	def __init__(self, fmt, datefmt=None):
		"""
		Init given the log line format and date format
		"""
		logging.Formatter.__init__(self, fmt, datefmt)

	def format(self, record):
		"""
		Override format function
		"""
		msg = logging.Formatter.format(self, record)

		if record.message != "":
			parts = msg.split(record.message)
			msg = msg.replace('\n', '\n' + parts[0])

		return msg


def load_logger(name):
	logger = logging.getLogger(name)
	logger.propagate = False
	formatter = NewLineFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

	return logger
