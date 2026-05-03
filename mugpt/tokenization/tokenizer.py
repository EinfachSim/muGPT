from abc import ABC, abstractmethod
import tiktoken
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class BaseTokenizer(ABC):
    @abstractmethod
    def encode_ordinary(self, text: str) -> list[int]:
        ...
    
    @property
    @abstractmethod
    def n_vocab(self) -> int:
        ...
    
    @property
    @abstractmethod
    def eot_token(self) -> int:
        ...
    
    def fit(self, corpus, n_vocab, out_path, text_column=None):
        pass

    @abstractmethod
    def load_from_file(self, path):
        ...
    
    @abstractmethod
    def encode(self, text):
        ...

class GPT2Tokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self.tok = tiktoken.get_encoding("gpt2")
    
    def encode_ordinary(self, text):
        return self.tok.encode_ordinary(text)
    
    @property
    def eot_token(self):
        return self.tok.eot_token
    
    @property
    def n_vocab(self):
        return self.tok.n_vocab
    
    def load_from_file(self, path):
        return self

    def encode(self, text):
        return self.tok.encode(text)
    
    def decode(self, ids):
        return self.tok.decode(ids)


class BPETokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()

        self.tok = Tokenizer(BPE())
        self.tok.pre_tokenizer = ByteLevel()
    
    @property
    def n_vocab(self):
        return self.tok.get_vocab_size()
    
    @property
    def eot_token(self):
        return self.tok.token_to_id("[EOT]")
    
    def fit(self, dataset, n_vocab, out_path, text_column="text", batch_size=1000):

        def batch_iterator():
            # Only keep the text column to avoid decoding the rest of the columns unnecessarily
            tok_dataset = dataset.select_columns(text_column)
            for batch in tok_dataset.iter(batch_size):
                yield batch["text"]
        
        trainer = BpeTrainer(
            vocab_size=n_vocab,
            special_tokens=["[UNK]", "[EOT]"],
        )

        self.tok.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))
        self.tok.save(out_path)
    
    def encode_ordinary(self, text):
        return self.tok.encode(text).ids
    
    def load_from_file(self, path):
        self.tok = Tokenizer.from_file(path)
        return self
    
    def encode(self, text):
        return self.tok.encode(text).ids
    
    def decode(self, ids):
        self.tok.decoder = ByteLevelDecoder()
        return self.tok.decode(ids)

    