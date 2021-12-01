import os
import sys
import ast
from typing import List


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.append(BASE_DIR)
print(__file__.split("/")[-1] + ": " + BASE_DIR)


from common.dialog import Dialog
from common.knowledge import Knowledge
from common.utter import Utter


class Data:
	def __init__(self, dialogs: List[Dialog]):
		self.dialogs: List[Dialog] = dialogs

	@classmethod
	def load_from_kvr_file(cls, file: str):
		file = os.path.join(DATA_DIR, file)
		dialogs: List[Dialog] = []
		knows: List[str] = []
		utters: List[Utter] = []
		with open(file, "r", encoding="utf-8") as f:
			line = f.readline()
			while line:
				line = line.strip()
				if line:
					if "#" in line:
						domain: str = line.replace("#", "")
					else:
						nid, item = line.split(" ", 1)
						if nid == "0":
							knows.append(item)
						else:
							assert "\t" in item
							user_sent, sys_sent, kb_mention = item.split("\t")
							user_utter = Utter(role="user", sent=user_sent, turn=int(nid))
							kb_mention = ast.literal_eval(kb_mention) # "['dentist_appointment']" -> ['dentist_appointment']
							sys_utter = Utter(role="sys", sent=sys_sent, kb_mention=kb_mention, turn=int(nid))
							utters.append(user_utter)
							utters.append(sys_utter)		
				else:
					knowledge = Knowledge(knows)
					dialog = Dialog(utters, knowledge, domain)
					knows = []
					domain = ""
					utters = []
					dialogs.append(dialog)
				line = f.readline()
		return cls(dialogs)