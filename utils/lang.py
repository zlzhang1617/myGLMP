import os
import sys
import ast
from typing import List, Dict, Union


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.append(BASE_DIR)
print(__file__.split("/")[-1] + ": " + BASE_DIR)


from utils.functions import add_word


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


class Lang:
	def __init__(self, idx2word: List[str]):
		self.idx2word: List[str] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
		for word in idx2word:	
			if word not in self.idx2word: self.idx2word.append(word)
		self.word2idx: Dict[str, int] = {word: idx for idx, word in enumerate(self.idx2word)}
		self.pad_token = PAD_TOKEN
		self.pad_token_idx = self.word2idx[PAD_TOKEN]
		self.unk_token = UNK_TOKEN
		self.unk_token_idx = self.word2idx[UNK_TOKEN]
		self.bos_token = BOS_TOKEN
		self.bos_token_idx = self.word2idx[BOS_TOKEN]
		self.eos_token = EOS_TOKEN
		self.eos_token_idx = self.word2idx[EOS_TOKEN]

	@classmethod
	def load_from_KVR(cls, files: List[str]):
		sketch_tags: List[str] = ["@event", "@time", "@date", \
									"@party", "@room", "@agenda", \
									"@location", "@weekly_time", "@temperature",\
									"@weather_attribute", "@traffic_info", \
									"@poi_type", "@poi", "@address", "$$$$",\
									"$u", "$s"]
		# generate special token id such as `turn:0`, `word_idx:0`, ...
		sketch_tags = sketch_tags + ["turn:"+str(i) for i in range(100)]
		sketch_tags = sketch_tags + ["word_idx:"+str(i) for i in range(100)]
		words: List[str] = []
		words += sketch_tags
		for file in files:
			file_name = os.path.join(DATA_DIR, file)
			with open(file_name, "r") as f:
				line = f.readline()
				while line:
					line = line.strip()
					if line:
						if "#" in line:
							line = line.replace("#", "")
							add_word(words, line)
							line = f.readline()
							continue
						id, dialog_sent = line.split(" ", 1)
						if id != "0":
							user_utter, sys_utter, kb_mention = dialog_sent.split("\t")
							add_word(words, user_utter.split(" "))
							add_word(words, sys_utter.split(" "))
							kb_mention = ast.literal_eval(kb_mention)
							add_word(words, kb_mention)
						else:
							add_word(words, dialog_sent.split(" "))
						line = f.readline()
					else:
						line = f.readline()
		
		return cls(words)


	def tokenize(self, sent: Union[str, List[str]], split_token: str=" ", add_bos: bool=True, add_eos: bool=True) -> List[str]:
		if isinstance(sent, str):
			tokens = sent.split(split_token)
			if add_bos: tokens = [self.bos_token] + tokens
			if add_eos: tokens = tokens + [self.eos_token]
		else:
			tokens = sent
			if add_bos: tokens = [self.bos_token] + tokens
			if add_eos: tokens = tokens + [self.eos_token]
		return tokens

	def encode(self, tokens: List[str]) -> List[int]:
		encoded_tokens = [self.word2idx.get(token, self.unk_token_idx) for token in tokens]
		return encoded_tokens

	def decode(self, encoded_tokens: List[int]) -> List[str]:
		tokens: List[str] = [self.idx2word[idx] for idx in encoded_tokens]
		return tokens