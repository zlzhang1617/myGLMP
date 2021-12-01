import json
import pickle
import copy
import os

from typing import Dict, Any, Union, List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
print(__file__.split("/")[-1] + ": " + BASE_DIR)


def read_json(file: str) -> Any:
	with open(file, "r") as f:
		data = json.load(f)
	return data


def write_json(file: str, data: Union[Dict[Any, Any], List[Any]]):
	with open(file, "w") as f:
		json.dump(data, f, ensure_ascii=False, indent=4)

def write_pickle(file: str, data: Any):
	with open(file, "wb") as f:
		pickle.dump(data, f)

def load_pickle(file: str):
	with open(file, "rb") as f:
		data = pickle.load(f)
	return data


def add_word(words: List[str], word: Union[str, List[str]]):
	if isinstance(word, str):
		if word not in words: words.append(word)
	else:
		for w in word:
			if w not in words: words.append(w)


kvr_entities = None
def generate_sketch_sent_by_kvr_entities(sent: str, kb_mention: List[str]) -> str:
	global kvr_entities
	if kvr_entities is None:
		kvr_entities = read_json(os.path.join(DATA_DIR, "kvret_entities.json"))
	if len(kb_mention) == 0: 
		return copy.deepcopy(sent)
	else:
		sketch_response: List[str] = []
		tokens = sent.split(" ")
		for word in tokens:
			if word not in kb_mention:
				sketch_response.append(word)
			else:
				entity_type = ""
				for k in kvr_entities.keys():
					if k != "poi":
						kvr_entities[k] = [x.lower() for x in kvr_entities[k]]
						if word in kvr_entities[k] or word.replace("_", " ") in kvr_entities[k]:
							entity_type = k
							sketch_response.append("@" + entity_type)
							break
					else:
						pois = [d["poi"].lower() for d in kvr_entities["poi"]]
						if word in pois or word.replace("_", " ") in pois:
							entity_type = "poi"
							sketch_response.append("@" + entity_type)
							break
						poi_addrs = [d["address"].lower() for d in kvr_entities["poi"]]
						if word in poi_addrs or word.replace("_", " ") in poi_addrs:
							entity_type = "address"
							sketch_response.append("@" + entity_type)
							break
		assert len(tokens) == len(sketch_response)
		return " ".join(sketch_response)


def get_global_entities() -> List[str]:
	global kvr_entities
	if kvr_entities is None:
		kvr_entities = read_json(os.path.join(DATA_DIR, "kvret_entities.json"))
	global_entities = []
	for key in kvr_entities.keys():
		if key != 'poi':
			global_entities += [item.lower().replace(' ', '_') for item in kvr_entities[key]]
		else:
			for item in kvr_entities['poi']:
				global_entities += [item[k].lower().replace(' ', '_') for k in item.keys()]
	global_entities: List[str] = list(set(global_entities))
	return global_entities


def generate_memory(sent: str, role: str, turn: int) -> List[List[str]]:
	sent_tokens = sent.split(" ")
	memory_rep: List[List[Any]] = []
	if role == "user" or role == "sys":
		role_s = "$u" if role == "user" else "$s"
		for idx, word in enumerate(sent_tokens):
			tmp = [role_s, "turn:"+str(turn), "word_idx:"+str(idx), word]
			memory_rep.append(tmp)
	else:
		# kb memory
		memory_rep.append(sent_tokens)
	return memory_rep


def pad_tokens(tokens: List[Any], pad_to_len: int, pad_token: Any) -> List[Any]:
	if len(tokens) < pad_to_len:
		tokens_ = tokens + [pad_token] * (pad_to_len - len(tokens))
	else:
		tokens_ = tokens[: pad_to_len]
	return tokens_

smooth_func = SmoothingFunction()
def nltk_multi_bleu(refs: List[List[str]], cands: List[str]) -> float:
	bleu = sentence_bleu(refs, cands, smoothing_function=smooth_func.method1)
	return bleu


def compute_entity_PRF(pred_resp: List[str], gold_entities: List[str], entities: List[str]):
	tp = 0 # True Postive. Where gold entity in predict response
	fp = 0 # False Postive. Entities not in gold list but in global entity dict which from predict response
	fn = 0 # False Negtive. Where gold entity not in predict response
	if len(gold_entities)!= 0:
		count = 1
		for g in gold_entities:
			if g in pred_resp:
				tp += 1
			else:
				fn += 1
		for p in set(pred_resp):
			if p in entities:
				if p not in gold_entities:
					fp += 1
		precision = tp / float(tp + fp) if (tp + fp) !=0 else 0
		recall = tp / float(tp + fn) if (tp + fn) !=0 else 0
		f1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
	else:
		precision, recall, f1, count = 0, 0, 0, 0
	return precision, recall, f1, count