import os
import sys
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import Optional, Tuple, List, Dict, Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
print(__file__.split("/")[-1] + ": " + BASE_DIR)
sys.path.append(BASE_DIR)


from utils.functions import load_pickle, write_pickle, generate_memory, pad_tokens
from utils.lang import Lang


class GLMP(nn.Module):
    def __init__(self, lang: Lang, hidden_size: int, \
                        memory_hop: int=3, dropout: float=0.2, \
                        sent_len: int=100):
        super(GLMP, self).__init__()
        self.lang = lang
        self.vocab_size = len(lang.idx2word) 
        self.hidden_size = hidden_size
        self.memory_hop = memory_hop
        self.dropout = dropout
        self.sent_len = sent_len

        self.share_mebed = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)

        self.external_knowledge = External_Knowledge(self.vocab_size, hidden_size, memory_hop, dropout)
        self.global_memory_encoder = Global_Memory_Encoder(self.share_mebed, hidden_size, dropout)
        self.local_memory_decoder = Local_Memory_Decoder(self.lang, self.share_mebed, self.hidden_size, self.dropout)

        self.loss_g = nn.BCELoss()
        self.loss_v = nn.CrossEntropyLoss(ignore_index=0)
        self.loss_l = nn.CrossEntropyLoss(ignore_index=-100)
    
        self.copy_list: List[List[str]] = []

    @classmethod
    def create_model(cls, model_name: str, device: str = "cpu"):
        model_path = os.path.join(MODEL_DIR, model_name)
        config_file = os.path.join(model_path, "config.pkl")
        model_file = os.path.join(model_path, "model.bin")
        if not os.path.exists(config_file) or not os.path.exists(model_file):
            print("please check model path: %s, config path: %s" % (model_file, config_file))
            raise FileNotFoundError
        config = load_pickle(config_file)
        model = cls(config["lang"], config["hidden_size"], config["memory_hop"], \
                    config["dropout"], config["sent_len"])
        model.to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        return model

    def get_model_info(self) -> Dict[str, Any]:
        info = {"lang": self.lang, "hidden_size": self.hidden_size, \
            "memory_hop": self.memory_hop, "dropout": self.dropout, \
            "sent_len": self.sent_len}
        return info

    def save_model(self, model_name: str) -> str:
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path): os.mkdir(model_path)
        torch.save(self.state_dict(), os.path.join(model_path, "model.bin"))
        config_path = os.path.join(model_path, "config.pkl")
        model_info = self.get_model_info()
        write_pickle(config_path, model_info)
        return model_path

    def forward(self, memory: torch.Tensor, dialog_memory: torch.Tensor, \
                        kb_len: torch.Tensor, dialog_len: torch.Tensor, resp_len: torch.Tensor, \
                        sketch_response: torch.Tensor, \
                        global_point_label: Optional[torch.Tensor]=None, \
                        local_point_label: Optional[torch.Tensor]=None):
        '''
            Args:
                memory              : [batch, memory_size, memory_len]
                dialog_memory       : [batch, dialog_len, memory_len]
                kb_len              : [batch]
                dialog_len          : [batch]
                resp_len            : [batch]
                global_point_label  : [batch, memory_size]
                sketch_response     : [batch, seq]
                local_point_label   : [batch, seq]
            Return:
                loss                : Tensor or None
                generate_seq        : [batch, seq]
        '''
        dialog_hidden, h_n = self.global_memory_encoder.forward(dialog_memory, dialog_len)
        init_query = h_n
        global_point, memory_query = self.external_knowledge.load_memory(memory, kb_len, dialog_len, dialog_hidden, init_query)
        # global_point: [batch, memory_size]
        # memory_query: [batch, hidden]

        # h_n: [batch, hidden]
        # memory_query: [batch, hidden]
        h_0 = torch.cat([h_n, memory_query], dim=-1) # [batch, hidden*2]
        local_point, seq_logits, = self.local_memory_decoder.forward(self.external_knowledge, \
                                                                memory, \
                                                                h_0, \
                                                                global_point, \
                                                                sketch_response)
        # local_point: [batch, seq, memory_size]
        # seq_logits: [batch, seq, vocab_size]
        memory_size = local_point.size(2)
        vocab_size = seq_logits.size(2)

        loss = None
        if global_point_label is not None and local_point_label is not None:
            loss_g = self.loss_g(global_point, global_point_label)
            loss_v = self.loss_v(seq_logits[:, :-1, :].contiguous().view(-1, vocab_size), sketch_response[:, 1:].contiguous().view(-1))
            loss_l = self.loss_l(local_point.view(-1, memory_size), local_point_label.view(-1))
            loss = loss_g + loss_v + loss_l

        return loss

    def predict(self, kb: List[List[str]], history: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        device = next(self.parameters()).device
        memorys: List[List[str]] = []
        memorys += kb
        kb_len = len(memorys)

        dialog_memory: List[List[str]] = []
        for idx, u_s in enumerate(history):
            role, sent = u_s
            sub_mem = generate_memory(sent, role, idx)
            dialog_memory += sub_mem
        dialog_len = len(dialog_memory)

        memorys = memorys + dialog_memory + [["$$$$"]*6]
        memory_len = [len(memorys)]

        self.copy_list = []
        for _ in range(1):
            self.copy_list.append([mem[-1] for mem in memorys])

        for idx, mem in enumerate(memorys):
            memorys[idx] = pad_tokens(mem, 6, self.lang.pad_token)

        encoded_memorys = [self.lang.encode(mem) for mem in memorys]
        encoded_dialog_memorys = [self.lang.encode(mem) for mem in dialog_memory]

        memorys_t = torch.tensor(encoded_memorys, dtype=torch.long, device=device).unsqueeze(dim=0)
        dialog_memorys_t = torch.tensor(encoded_dialog_memorys, dtype=torch.long, device=device).unsqueeze(dim=0)
        kb_len_t = torch.tensor(kb_len, dtype=torch.long, device=device).unsqueeze(dim=0)
        dialog_len_t = torch.tensor(dialog_len, dtype=torch.long, device=device).unsqueeze(dim=0)

        dialog_hidden, h_n = self.global_memory_encoder.forward(dialog_memorys_t, dialog_len_t)
        init_query = h_n
        global_point, memory_query = self.external_knowledge.load_memory(memorys_t, kb_len_t, dialog_len_t, dialog_hidden, init_query)

        h_0 = torch.cat([h_n, memory_query], dim=-1) # [batch, hidden*2] 
        coarse_sents, fine_sents = self.local_memory_decoder.decode(self.external_knowledge, \
                                                                    memorys_t, \
                                                                    memory_len, \
                                                                    h_0, \
                                                                    global_point, \
                                                                    self.sent_len, \
                                                                    self.copy_list)

        return coarse_sents[0], fine_sents[0]


class External_Knowledge(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int,\
                        memory_hop: int=3, dropout: float=0.2):
        super(External_Knowledge, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.memory_hop = memory_hop
        self.dropout = dropout

        self.C = nn.ModuleList()
        for _ in range(memory_hop+1):
            e = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            e.weight.data.normal_(0, 0.1)
            self.C.append(e)
        self.dropout_layer = nn.Dropout(dropout)

    def add_hidden(self, embed_memory: torch.Tensor, kb_len: torch.Tensor, dialog_len: torch.Tensor, dialog_hidden: torch.Tensor):
        '''
        Args:
            embed_memory        : [batch, memory_size, embed]
            kb_len              : [batch]
            dialog_len          : [batch]
            dialog_hidden       : [batch, seq, hidden]
        '''
        bs = embed_memory.size(0)
        for idx in range(bs):
            start = kb_len[idx]
            end = start + dialog_len[idx]
            embed_memory[idx, start: end, :] = embed_memory[idx, start: end, :] + dialog_hidden[idx, :dialog_len[idx], :]
        return embed_memory

    def load_memory(self, memory: torch.Tensor, \
                        kb_len: torch.Tensor, dialog_len: torch.Tensor, \
                        dialog_hidden: torch.Tensor, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            memory            : [batch, memory_size, memory_len]
            kb_len            : [batch]
            dialog_len        : [batch]
            dialog_hidden     : [batch, seq_len, hidden]
            query             : [batch, hidden]
        Return:
            global_point      : [batch, memory_size]
            q_k               : [batch, embed]
        '''
        self.memorys = []
        u = [query]
        for k in range(self.memory_hop):
            embed_m = self.C[k](memory) # [batch, memory_size, memory_len, embed]
            embed_m = torch.sum(embed_m, dim=2) # [batch, memory_size, embed]
            embed_m = self.add_hidden(embed_m, kb_len, dialog_len, dialog_hidden) # [batch, memory_size, embed]
            embed_m = self.dropout_layer(embed_m)

            # calculate score between query and m
            q_k = u[-1] # [batch, hidden]
            q_k_tmp = q_k.unsqueeze(dim=1).expand_as(embed_m) # [batch, memory_size, embed]
            logits = torch.sum(embed_m * q_k_tmp, dim=-1) 
            # logits = torch.sum(q_k_tmp * embed_m, dim=-1) 
            probs = torch.softmax(logits, dim=-1) # [batch, memory_size]

            embed_c = self.C[k+1](memory) # [batch, memory_size, memory_len, embed]
            embed_c = torch.sum(embed_c, dim=2) # [batch, memory_size, embed]
            embed_c = self.add_hidden(embed_c, kb_len, dialog_len, dialog_hidden) # [batch, memory_size, embed]
            embed_c = self.dropout_layer(embed_c)

            # probs * c
            probs_ = probs.unsqueeze(dim=2).expand_as(embed_c)
            o_k = torch.sum(embed_c * probs_, dim=1) # [batch, embed]
            # o_k = torch.sum(probs_ * embed_c, dim=1) # [batch, embed]

            u.append(u[-1] + o_k)
            self.memorys.append(embed_m)

        self.memorys.append(embed_c)

        q_k = u[-1]
        global_point = torch.sigmoid(logits)
        return global_point, q_k

    def forward(self, global_point: torch.Tensor, query: torch.Tensor):
        '''
        Args:
            global_point        : [batch, memory_size]
            query               : [batch, hidden]
        Return:
            probs               : [batch, memory_size]
        '''
        u = [query]
        for k in range(self.memory_hop):
            q_k = u[-1]
            embed_A = self.memorys[k] # [batch, memory_size, hidden]
            embed_A = embed_A * global_point.unsqueeze(dim=2).expand_as(embed_A)

            q_tmp = q_k.unsqueeze(dim=1).expand_as(embed_A) # [batch, memory_size, hidden]
            logits = torch.sum(embed_A * q_tmp, dim=2)
            # logits = torch.sum(q_tmp * embed_A, dim=2)
            probs = torch.softmax(logits, dim=-1) # [batch, memory_size]

            embed_C = self.memorys[k+1]
            embed_C = embed_C * global_point.unsqueeze(dim=2).expand_as(embed_C)

            p_tmp = probs.unsqueeze(dim=2).expand_as(embed_C) # [batch, memory_size, hidden]
            o_k = torch.sum(embed_C * p_tmp, dim=1)
            # o_k = torch.sum(p_tmp * embed_C, dim=1)
            u.append(u[-1] + o_k)
        return probs, logits


class Global_Memory_Encoder(nn.Module):
    def __init__(self, shared_embed: nn.Embedding, hidden_size: int, dropout: float):
        super(Global_Memory_Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.embedding = shared_embed
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, dialog_memory: torch.Tensor, dialog_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            dialog_memory         : [batch, seq_len, memory_len]
            dialog_len            : [batch]
        Return:
            dialog_hiddens        : [batch, seq_len, hidden]
            h_n                   : [batch, hidden]
        '''
        embed_dialog = self.embedding(dialog_memory) # [batch, seq_len, memory_len, embed]
        embed_dialog = torch.sum(embed_dialog, dim=2) # [batch, seq_len, embed]
        embed_dialog = self.dropout_layer(embed_dialog)
        paded_dialog = pack_padded_sequence(embed_dialog, dialog_len, batch_first=True, enforce_sorted=False)

        outputs, h_n = self.gru(paded_dialog)
        # h_n: [2, batch, hidden]
        h_n = torch.cat([h_n[0], h_n[1]], dim=-1) # [batch, hidden*2]
        h_n = self.linear(h_n)

        hiddens, _ = pad_packed_sequence(outputs, batch_first=True)
        dialog_hiddens = self.linear(hiddens)
        return dialog_hiddens, h_n


class Local_Memory_Decoder(nn.Module):
    def __init__(self, lang: Lang, share_embed: nn.Embedding, hidden_size:int, \
                        dropout: float=0.2):
        super(Local_Memory_Decoder, self).__init__()
        self.lang = lang
        self.bos_idx = self.lang.bos_token_idx
        self.pad_idx = self.lang.pad_token_idx
        self.vocab_size = len(self.lang.word2idx)
        self.hidden_size = hidden_size

        self.embedding = share_embed
        self.sketch_rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, \
                                    num_layers=1, bidirectional=False)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, ext_know: External_Knowledge, memory: torch.Tensor, \
                        h_0: torch.Tensor, global_point: torch.Tensor, \
                        target_resp: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            ext_know            : instance of class `External Knowledge`
            memory              : [batch, memory_size, memory_len]
            h_0                 : [batch, hidden*2]
            target_resp         : [batch, seq]
            global_point        : [batch, memory_size]
        Return:
            local_memory_point  : [batch, seq, memory_size]
            seq_logits          : [batch, seq, vocab]
        '''
        bs = h_0.size(0)
        memory_size = memory.size(1)
        device = next(self.parameters()).device
        seq_len = target_resp.size(1)
        
        decode_seq = torch.zeros([1, bs], device=device, dtype=torch.long).fill_(self.bos_idx) # [seq, batch]
        seq_logits = torch.zeros([seq_len, bs, self.vocab_size], device=device, dtype=torch.float)
        local_memory_point = torch.zeros([seq_len, bs, memory_size], device=device, dtype=torch.float)

        last_hidden = torch.relu(self.linear(h_0)).unsqueeze(dim=0) # [1, batch, hidden]
        force_training_flag = True if random.random() < 0.5 else False
        for step_idx in range(seq_len-1):
            decode_seq_embed = self.dropout_layer(self.embedding(decode_seq)) # [seq, batch, hidden]
            _, h_n = self.sketch_rnn(decode_seq_embed, last_hidden)
            last_hidden = h_n # [1, batch, hidden]

            vocab_logits = self.attend_vocab(self.embedding.weight, h_n.squeeze(dim=0)) # [batch, vocab]
            topvi = vocab_logits.argmax(dim=-1, keepdim=True) # [batch, 1]
            seq_logits[step_idx] = vocab_logits

            query_vec = h_n.squeeze(dim=0)
            _, memory_logits = ext_know.forward(global_point, query_vec) # [batch, memory_size]
            
            local_memory_point[step_idx] = memory_logits

            if force_training_flag:
                decode_seq = target_resp[:, step_idx+1].unsqueeze(dim=0)
            else:
                decode_seq = topvi.transpose(0, 1)
        
        local_memory_point: torch.Tensor = local_memory_point.transpose(0, 1).contiguous()
        seq_logits: torch.Tensor = seq_logits.transpose(0, 1).contiguous()

        return local_memory_point, seq_logits

    def decode(self, ext_know: External_Knowledge, \
                    memory: torch.Tensor, memory_len: List[int],\
                    h_0: torch.Tensor, global_point: torch.Tensor, max_sent_lens: int, \
                    copy_list: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        '''
        Args:
            ext_know        : Instance of class `External_Knowledge`
            memory          : [batch, memory_size, memory_len]
            memory_len      : List[batch]
            h_0             : [batch, 2*hidden] # bidirectional GRU
            global_point    : [batch, memory_size]
            max_sent_lens   : int
            copy_list       : List[List[str]] # memory entities and history words
        Return:
            coarse_sent     : List[List[str]] # resp with special tokens
            fine_sent       : List[List[str]] # resp without special tokens
        '''
        bs = h_0.size(0)
        memory_size = memory.size(1)
        device = next(self.parameters()).device

        decode_seq = torch.zeros([1, bs], device=device, dtype=torch.long).fill_(self.bos_idx) # [seq, batch]
        memory_mask = torch.ones([bs, memory_size], dtype=torch.float, device=device)

        coarse_sent: List[List[str]] = [[] for _ in range(bs)]
        fine_sent: List[List[str]] = [[] for _ in range(bs)]

        last_hidden = torch.relu(self.linear(h_0)).unsqueeze(dim=0) # [1, batch, hidden]
        for _ in range(max_sent_lens):
            decode_seq_embed = self.dropout_layer(self.embedding(decode_seq)) # [1, batch, hidden]
            _, h_n = self.sketch_rnn(decode_seq_embed, last_hidden)
            last_hidden = h_n # [1, batch, hidden]
            query_vec = h_n[0]

            memory_probs, _ = ext_know.forward(global_point, query_vec) # [batch, memory_size]
            search_lens = min(5, min(memory_len))
            memory_probs = memory_probs * memory_mask
            _, toppi = memory_probs.data.topk(search_lens)

            vocab_logits = self.attend_vocab(self.embedding.weight, h_n.squeeze(dim=0))
            vocab_probs = torch.softmax(vocab_logits, dim=-1) # [batch, vocab_size]

            pick_words_t = torch.argmax(vocab_probs, dim=-1, keepdim=True) # [batch, 1]
            decode_seq = pick_words_t.transpose(0, 1) # [1, batch]

            pick_words_idx = pick_words_t.cpu().numpy().tolist()

            set_words = [i for item in pick_words_idx for i in item]
            if set(set_words) == set([self.lang.eos_token_idx, self.lang.pad_token_idx]):
                return coarse_sent, fine_sent
            
            for bidx in range(bs):
                pick_words_idx_ = pick_words_idx[bidx]
                pick_words = self.lang.decode(pick_words_idx_)
                for word in pick_words:
                    if word in [self.lang.pad_token, self.lang.eos_token]:
                        continue
                    coarse_sent[bidx].append(word)
                    if "@" in word:
                        for i in range(search_lens):
                            if toppi[bidx, i] < memory_len[bidx] - 1:
                                fine_sent[bidx].append(copy_list[bidx][toppi[bidx, i]])
                                memory_mask[bidx, toppi[bidx, i].item()] = 0
                                break
                    else:
                        fine_sent[bidx].append(word)

        return coarse_sent, fine_sent

    def attend_vocab(self, seq: nn.Parameter, cond: torch.Tensor):
        '''
        Args:
            seq     : [vocab, hidden]
            cond    : [batch, hidden]
        Return:
            score   : [batch, vocab]
        '''
        score = cond.matmul(seq.transpose(0,1))
        return score