# GLMP论文复现

## 使用

项目文件夹（myGLMP）放在家目录如 `/home/zzl/myGLMP`

训练命令

```bash
 bash GLMP/scripts/train.sh
```

测试命令

```bash
 bash GLMP/scripts/inference.sh <modelname>
# bash GLMP/scripts/inference.sh K3
```

运行结果在 `GLMP/log/`目录

## 项目说明

* `common` 目录存放抽象类
* `data` 目录存放SMD数据集
* `GLMP` 目录存放模型和dataset相关
* `models` 目录用于存放训练好的模型
* `utils`
  * `functions.py` 存放工具函数
  * `lang.py` 用于字词编码

### 数据集

SMD数据集格式见 `data/train.txt`，例如：

> #schedule#
> 0 tennis_activity date the_8th
> 0 tennis_activity party brother
> 1 what is the date and time for the yoga_activity ?	you have two yoga activites scheduled one on wednesday at 5pm and another on the_17th at 11am	['11am', 'yoga', '5pm', 'the_17th', 'wednesday']
> 2 thanks	glad i could help , goodbye	[]

第一行#schedule#是该数据样例所属domain

行首数字0表示knowledge

数字1+表示第 `i`轮对话，每一轮对话由换行符 `\t`分割，分别是 `user`、`system`、`entity`

### 模型

GLMP由 `Global Memory Encoder`、`External Knowledge`、`Local Memory Decoder`三部分组成，其中 `Global Memory Encoder`, `Local Memory Decoder`共享embedding层
详见项目 `GLMP/model.py`文件

#### Global Memory Encoder

该模块由embedding和GRU组成

```python
class Global_Memory_Encoder(nn.Module):
    def __init__(self, shared_embed: nn.Embedding, hidden_size: int):
        super(Global_Memory_Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = shared_embed
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=1, bidirectional=True)
```

与一般的文本序列不同，作者将对话历史处理为n元组格式，n元组是 [`$u or $s`, `turn idx`, `word idx`, `word`]

例如：

>
> 对话--------
> user: what is the date and time for the yoga_activity ?
> system: you have two yoga activites scheduled one on wednesday at 5pm and another on the_17th at 11am
>
> n元组-------
> [
> ["$u", "turn0", "word0", "what"],
> ["$u", "turn0", "word1", "is"],
> ["$u", "turn0", "word2", "the"],
> ...
> ["$s", "turn0", "word27", "11am"],
> ]

输入n元组的定长编码，输出序列hidden和对话hidden

```python
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
```

#### External Knowledge

`External Knowledge`模块使用记忆网络（Memory Network）存储和访问knowledge（数据集中标号为0的行）
knowledge编码为形状 `[batch, memory_size, memory_len]`的memory矩阵

对于一个有$k$层的Memory Network(k-hop)，它由$k+1$个可训练的embedding层组成。

```python
class External_Knowledge(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, memory_hop: int=3):
        super(External_Knowledge, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.memory_hop = memory_hop

        self.C = nn.ModuleList()
        for _ in range(memory_hop+1):
            e = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            e.weight.data.normal_(0, 0.1)
            self.C.append(e)
```

可分为 `load memory`和 `search memory`两个步骤

##### 1. load memory

初始一个查询向量$q$和memory矩阵
$q_0 \in R^{[batch, hidden]}$
$memory \in R^{[batch, memory_size, memory_len]}$

在MN的第$i$层，用第$i$和$i+1$个embedding将$memory$映射为$embedA$和$embedC$

用查询向量$q_i$与$embedA$做点积得到Attention score

Attention score作为$embedC$的权重，对$embedC$做带权求和得到向量$o_i$

$o_i + q_i$更新查询向量，作为下一层MN的初始查询向量$q_{i+1}$

将第$i$层的$embedA$和最后第$k$层的$embedC$记录下来，作为存储的Memory

在该任务中对 `memory_len`维度做加和操作来保证张量大小一致。

```python
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
            probs = torch.softmax(logits, dim=-1) # [batch, memory_size]

            embed_c = self.C[k+1](memory) # [batch, memory_size, memory_len, embed]
            embed_c = torch.sum(embed_c, dim=2) # [batch, memory_size, embed]
            embed_c = self.add_hidden(embed_c, kb_len, dialog_len, dialog_hidden) # [batch, memory_size, embed]
            embed_c = self.dropout_layer(embed_c)

            # probs * c
            probs_ = probs.unsqueeze(dim=2).expand_as(embed_c)
            o_k = torch.sum(embed_c * probs_, dim=1) # [batch, embed]

            u.append(u[-1] + o_k)
            self.memorys.append(embed_m)

        self.memorys.append(embed_c)

        q_k = u[-1]
        global_point = torch.sigmoid(logits)
        return global_point, q_k
```

##### 2. search memory

输入初始查询向量$q$，输出memory的概率分布

过程与 `1.`相同，但$embedA$和$embedC$来自于 `1.`中记录的memory

```python
def forward(self, global_point: torch.Tensor, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            global_point        : [batch, memory_size]
            query               : [batch, hidden]
        Return:
            probs               : [batch, memory_size]
			logits				: [batch, memory_size]
        '''
        u = [query]
        for k in range(self.memory_hop):
            q_k = u[-1]
            embed_A = self.memorys[k] # [batch, memory_size, hidden]
            embed_A = embed_A * global_point.unsqueeze(dim=2).expand_as(embed_A)

            q_tmp = q_k.unsqueeze(dim=1).expand_as(embed_A) # [batch, memory_size, hidden]
            logits = torch.sum(embed_A * q_tmp, dim=2)
            probs = torch.softmax(logits, dim=-1) # [batch, memory_size]

            embed_C = self.memorys[k+1]
            embed_C = embed_C * global_point.unsqueeze(dim=2).expand_as(embed_C)

            p_tmp = probs.unsqueeze(dim=2).expand_as(embed_C) # [batch, memory_size, hidden]
            o_k = torch.sum(embed_C * p_tmp, dim=1)
            u.append(u[-1] + o_k)
        return probs, logits
```

#### Local Memory Decoder

`Local Memory Decoder`由 `share embed`、`GRU`组成，用自回归的方式生成句子

```python
class Global_Memory_Encoder(nn.Module):
    def __init__(self, shared_embed: nn.Embedding, hidden_size: int):
        super(Global_Memory_Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = shared_embed
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, hidden_size)
```

该模块生成的是sketch response，是带有特殊符号的句子。例如"go to @address @time"

并对特殊符号字符进一步通过 `External Knowledge`完成替换，得到完整回复"go to scool tomorrow"

自回归过程：

```python
decode_seq = torch.zeros([1, bs], device=device, dtype=torch.long).fill_(self.bos_idx) # [seq, batch]
seq_logits = torch.zeros([seq_len, bs, self.vocab_size], device=device, dtype=torch.float)
seq_len = 50 # 超参数
last_hidden # [1, batch, hidden] 从Encoder编码得到
for step_idx in range(seq_len-1):
	decode_seq_embed = self.dropout_layer(self.embedding(decode_seq)) # [seq, batch, hidden]
	_, h_n = self.sketch_rnn(decode_seq_embed, last_hidden)
	last_hidden = h_n # [1, batch, hidden]

	vocab_logits = self.attend_vocab(self.embedding.weight, h_n.squeeze(dim=0)) # [batch, vocab]
	topvi = vocab_logits.argmax(dim=-1, keepdim=True) # [batch, 1]
	seq_logits[step_idx] = vocab_logits

	decode_seq = topvi.transpose(0, 1)
```

自回归loss计算：

```python
loss_func = nn.CrossEntropyLoss(ignore_index=-100)
seq_logits # [batch, seq, vocab]
sketch_response # [batch, seq]
loss = loss_func(seq_logits[:, :-1, :].contiguous().view(-1, vocab_size), sketch_response[:, 1:].contiguous().view(-1))

# !! 
# 注意seq_logits第i位输出的概率分布，对应的是下一个i+1单词的正确分类
# 因此要取seq_logits[:, :-1, :] 与 sketch_reponse[:, 1:]对齐
```

解码过程：

```python
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
```
