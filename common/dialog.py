import os
import sys
from typing import List


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
print(__file__.split("/")[-1] + ": " + BASE_DIR)


from common.knowledge import Knowledge
from common.utter import Utter


class Dialog:
	def __init__(self, utters: List[Utter], knowledge: Knowledge, domain: str):
		self.utters = utters
		self.knowledge = knowledge
		self.domain = domain