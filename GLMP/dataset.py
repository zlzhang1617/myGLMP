import os
import sys
import copy
import torch
from typing import Any, List, Dict
from tqdm import tqdm
from torch.utils.data import Dataset


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.append(BASE_DIR)
print(__file__.split("/")[-1] + ": " + BASE_DIR)


from utils.lang import Lang
from utils.functions import pad_tokens
from common.data import Data


class GLMP_Dataset(Dataset):
	def __init__(self, lang: Lang, data: Data, memory_len: int=6, memory_size: int=500, sent_len: int=100):
		self.lang = lang
		self.memory_len = memory_len
		self.sent_len = sent_len
		self.memory_size = memory_size
		self.samples = self.generate_samples_from_data(data)
	
	def generate_samples_from_data(self, data: Data) -> List[Dict[str, Any]]:
		'''
		Args:
			data						: instance of class `Data`
		Return:
			samples						: []
		eg.
		sample in samples = {
			"history"					: []
			"response"					: ""
			"sketch_response"			: ""
			"dialog_memory"				: []
			"kb_memory"					: []
			"memory"					: []
			"memory_label"				: []
			"sketch_resp_point_label"	: []
			""
		}
		sample = {
			"history"					: ["where s the nearest parking_garage"]
			"response"					: "the nearest parking_garage is dish_parking at 550_alester_ave would you like directions there ?"
			"sketch_response"			: "the nearest @poi_type is @poi at @address would you like directions there ?"
			"dialog_memory"				: [["$u", "turn:0", "word:0", "where"], ["$u", "turn:0", "word:0", "s"], ...]
			"kb_memory"					: [["dish_parking", "distance", "2_miles"], ["dish_parking", "traffic_info", "road_block_nearby"], ...]
			"memory"					: [["dish_parking", "distance", "2_miles"], ..., ["where", "$u", "turn:0", "word:0"], ..., ["$$$$", "$$$$", "$$$$", "$$$$"]] 
			"memory_label"				: [0, 0, 0, ..., 1] # 1 or 0 which means whether memory position element occur in next response or not 
			"sketch_resp_point_label"	: [45, 45, ..., 45] # point to word's corresponding memory postion. no sketch tag point to index of $$$$
		}
		'''
		samples: List[Dict[str, Any]] = []
		dialogs = data.dialogs
		with tqdm(total=len(dialogs)) as pbar:
			pbar.set_description("loading")
			for dialog in dialogs:
				pbar.update(1)
				utters = dialog.utters

				dialog_memory: List[List[str]] = []
				kb_memory = dialog.knowledge.kb_memory
				kb_lens = len(kb_memory)
				history: List[str] = []
				memory: List[List[str]] = []
				memory_label: List[int] = []
				sketch_resp_point_label: List[int] = []

				for i in range(0, len(utters), 2):
					user_utter = utters[i]
					
					history.append(user_utter.sent)
					dialog_memory += user_utter.utter_memory

					memory = kb_memory + dialog_memory + [["$$$$"] * self.memory_len]

					sys_utter = utters[i+1]
					response = sys_utter.sent
					response_tokens = response.split(" ")
					sketch_response = sys_utter.sketch_sent
					sketch_response_tokens = sketch_response.split(" ")
					assert len(response_tokens) == len(sketch_response_tokens)

					memory_label:List[int] = [0] * len(memory)
					memory_label[-1] = 1
					assert len(memory) == len(memory_label)
					for idx, mem_triple in enumerate(memory[:-1]):
						if idx < kb_lens:
							obj = mem_triple[-1]
							if obj in sys_utter.kb_mention:
								memory_label[idx] = 1
						else:
							history_word = mem_triple[-1]
							if history_word in response:
								memory_label[idx] = 1

					sketch_resp_point_label = []
					null_idx = len(memory) - 1
					for word in response_tokens:
						label_idx = null_idx
						for idx, mem_triple in enumerate(memory):
							obj = mem_triple[-1]
							if obj == word and word in sys_utter.kb_mention:
								label_idx = idx
								break
						sketch_resp_point_label.append(label_idx)

					sample: Dict[str, Any] = {
						"history": copy.deepcopy(history),
						"response": response,
						"sketch_response": sketch_response,
						"dialog_memory": copy.deepcopy(dialog_memory),
						"kb_memory": copy.deepcopy(kb_memory),
						"memory": copy.deepcopy(memory),
						"memory_label": memory_label,
						"sketch_resp_point_label": sketch_resp_point_label
					}
					samples.append(sample)
					history.append(response)
					dialog_memory += sys_utter.utter_memory
		return samples

	def __getitem__(self, index: int):
		sample = self.samples[index]

		dialog_memory = sample["dialog_memory"]
		kb_memory = sample["kb_memory"]
		memory = sample["memory"]
		memory_label = sample["memory_label"]

		dialog_len = len(dialog_memory)
		kb_len = len(kb_memory)

		sketch_resp = sample["sketch_response"]
		sketch_resp_point_label = sample["sketch_resp_point_label"]

		# pad mememory
		## pad memory item
		for idx, mem in enumerate(dialog_memory):
			dialog_memory[idx] = pad_tokens(mem, self.memory_len, self.lang.pad_token)

		for idx, mem in enumerate(memory):
			memory[idx] = pad_tokens(mem, self.memory_len, self.lang.pad_token)

		# pad response
		sketch_tokens = self.lang.tokenize(sketch_resp)

		sketch_resp_point_label = [-100] + sketch_resp_point_label + [-100] # pad for <bos> and <eos>
		assert len(sketch_resp_point_label) == len(sketch_tokens)

		# encode memory
		encoded_memory = [self.lang.encode(mem) for mem in memory]
		encoded_dialog_memory = [self.lang.encode(mem) for mem in dialog_memory]

		# encode response
		encoded_sketch_tokens = self.lang.encode(sketch_tokens)
		resp_len = len(encoded_sketch_tokens)

		'''
		encoded_memory			: [memory_size, memory_len]
		encoded_dialog_memory	: [memory_size, memory_len]
		kb_len					: int
		dialog_len				: int
		memory_label			: [memory_size, memory_len]
		encoded_sketch_tokens	: [seq_len]
		sketch_resp_point_label	: [seq_len]
		'''
		return  encoded_memory, \
				encoded_dialog_memory, \
				kb_len, dialog_len, resp_len,\
				memory_label, \
				encoded_sketch_tokens, \
				sketch_resp_point_label, \
				

	def __len__(self) -> int:
		return len(self.samples)


def collate_fn(samples: List[Any]):
	'''
	Args:
		sample[i] = (
			memory				: [memory_size, memory_len]
			dialog_memory		: [memory_size, memory_len]
			kb_len				: int
			dialog_len			: int
			resp_len			: int
			memory_label		: [memory_size]
			sketch				: [seq_len]
			sketch_point_label	: [seq_len]
		)
	'''
	max_memory_size = max([len(sample[0]) for sample in samples])
	max_dialog_memory_size = max([len(sample[1]) for sample in samples])
	max_sent_len = max([sample[4] for sample in samples])

	memorys: List[List[List[int]]] = [] 			# [batch, max_memory_size, memory_len]
	dialog_memorys: List[List[List[int]]] = [] 		# [batch, max_dialog_memory_size, memory_len]
	kb_lens: List[int] = [] 						# [batch]
	dialog_lens: List[int] = [] 					# [batch]
	resp_lens: List[int] = []						# [batch]
	memory_labels: List[List[int]] = [] 			# [batch, max_memory_size]
	sketch_resps: List[List[int]] = [] 				# [batch, seq_len]
	sketch_resp_labels: List[List[int]] = [] 		# [batch, seq_len]
	

	bs = len(samples)
	for i in range(bs):
		sample = list(samples[i])

		memory = sample[0]
		memory_pad = [0] * len(memory[0])
		if len(memory) < max_memory_size:
			sample[0] = pad_tokens(memory, max_memory_size, memory_pad)
		else:
			sample[0] = sample[0][: max_memory_size]
		memorys.append(sample[0])

		dialog_memory = sample[1]
		if len(dialog_memory) < max_dialog_memory_size:
			sample[1] = pad_tokens(dialog_memory, max_dialog_memory_size, memory_pad)
		else:
			sample[1] = sample[1][: max_dialog_memory_size]
		dialog_memorys.append(sample[1])

		kb_lens.append(sample[2])
		dialog_lens.append(sample[3])
		resp_lens.append(sample[4])
		
		memory_label: List[Any] = sample[5]
		if len(memory_label) < max_memory_size:
			sample[5] = pad_tokens(memory_label, max_memory_size, 0)
		else:
			sample[5] = sample[5][: max_memory_size]
		memory_labels.append(sample[5])

		sketch_resp = sample[6]
		if len(sketch_resp) < max_sent_len:
			sample[6] = pad_tokens(sketch_resp, max_sent_len, 0)
		else:
			sample[6] = sample[6][: max_sent_len]
		sketch_resps.append(sample[6])
		
		sketch_resp_label = sample[7]
		if len(sketch_resp_label) < max_sent_len:
			sample[7] = pad_tokens(sketch_resp_label, max_sent_len, -100)
		else:
			sample[7] = sample[7][: max_sent_len]
		sketch_resp_labels.append(sample[7])

	return torch.tensor(memorys, dtype=torch.long),\
			torch.tensor(dialog_memorys, dtype=torch.long), \
			torch.tensor(kb_lens, dtype=torch.long), \
			torch.tensor(dialog_lens, dtype=torch.long), \
			torch.tensor(resp_lens, dtype=torch.long), \
			torch.tensor(memory_labels, dtype=torch.float), \
			torch.tensor(sketch_resps, dtype=torch.long), \
			torch.tensor(sketch_resp_labels, dtype=torch.long)


if __name__ == "__main__":
	lang = Lang.load_from_KVR(["train.txt", "dev.txt"])
	data = Data.load_from_kvr_file("train.txt")
	ds = GLMP_Dataset(lang, data)
	from torch.utils.data import DataLoader
	dl = DataLoader(dataset=ds, batch_size=2)
	batch = next(iter(dl))
	print("test")