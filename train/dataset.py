import torch
from torch.utils.data import Dataset
from utils import SmilesEnumerator
import numpy as np
import re

class SmileDataset(Dataset):

    def __init__(self, args, data, content, block_size, aug_prob = 0.5, prop = None, scaffold = None, scaffold_maxlen = None):
        '''
        Args:
            data:  smiles
            content: 词表
            block_size: int, the maximum length of  tokenized smiles sequences
            aug_prob: float, the probability of applying SMILES augmentation
            prop: pd.Series or np.ndarray, the properties corresponding to each smiles
            scaffold: pd.Series or np.ndarray, the scaffolds corresponding to each smiles
            scaffold_maxlen: int, the maximum length of scaffold sequences
        '''
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.debug = args.debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob
    
    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        smiles, prop, scaffold = self.data[idx], self.prop[idx], self.sca[idx]    # self.prop.iloc[idx, :].values  --> if multiple properties
        smiles = smiles.strip()
        scaffold = scaffold.strip()


        '''
        一个smiles分子式可能有多个有效的表示形式，SMILES增强技术通过随机化SMILES字符串的顺序来生成不同的表示形式，从而增加训练数据的多样性。aug_prob参数控制了应用SMILES增强的概率。
        '''
        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)

        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        smiles += str('<')*(self.max_len - len(regex.findall(smiles)))

        if len(regex.findall(smiles)) > self.max_len:
            smiles = smiles[:self.max_len]

        smiles=regex.findall(smiles)

        scaffold += str('<')*(self.scaf_max_len - len(regex.findall(scaffold)))
        
        if len(regex.findall(scaffold)) > self.scaf_max_len:
            scaffold = scaffold[:self.scaf_max_len]

        scaffold=regex.findall(scaffold)

        dix =  [self.stoi[s] for s in smiles]
        sca_dix = [self.stoi[s] for s in scaffold]

        sca_tensor = torch.tensor(sca_dix, dtype=torch.long)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.long)
        prop = torch.tensor([prop], dtype = torch.float)
        return x, y, prop, sca_tensor
