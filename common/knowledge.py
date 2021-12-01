import os
import sys
from typing import List


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
print(__file__.split("/")[-1] + ": " + BASE_DIR)


from utils.functions import generate_memory


class Knowledge:
	def __init__(self, knows: List[str]) -> None:
		self.kb_memory: List[List[str]] = []
		for know in knows:
			self.kb_memory += generate_memory(know, "", 0)