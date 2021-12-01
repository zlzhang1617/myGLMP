import os
import sys
from typing import List


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
print(__file__.split("/")[-1] + ": " + BASE_DIR)


from utils.functions import generate_sketch_sent_by_kvr_entities, generate_memory


class Utter:
	def __init__(self, role: str, sent: str, turn: int, kb_mention: List[str] = []):
		self.role = role
		self.sent = sent
		self.turn = turn
		self.kb_mention = kb_mention
		self.utter_memory = generate_memory(self.sent, self.role, self.turn)
		self.sketch_sent: str = generate_sketch_sent_by_kvr_entities(self.sent, self.kb_mention) if self.role == "sys" else sent