import sentencepiece as spm
import os
from typing import List, Optional

class SPTokenizer:
    """A SentencePiece tokenizer that mimics the tiktoken interface"""
    
    def __init__(self, model_path: str):
        """Initialize the tokenizer with a SentencePiece model file"""
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
    
    @classmethod
    def from_pretrained(cls, name: str, cache_dir: Optional[str] = None) -> 'SPTokenizer':
        """Load a pretrained SentencePiece model"""
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/sentencepiece")
        os.makedirs(cache_dir, exist_ok=True)
        """
        model_path = name  #os.path.join(cache_dir, f"{name}.model")
        if not os.path.exists(model_path):
            raise ValueError(f"Model {name} not found at {model_path}. Please train a SentencePiece model first.")
        
        return cls(model_path)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        return self.sp.EncodeAsIds(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token ids to text"""
        return self.sp.DecodeIds(tokens)
    
    def train(
        self,
        input_file: str,
        vocab_size: int = 32000,
        model_prefix: str = "sp_model",
        character_coverage: float = 0.9995,
        model_type: str = "bpe",
        user_defined_symbols: Optional[List[str]] = None,
    ):
        """Train a new SentencePiece model"""
        train_args = {
            "input": input_file,
            "model_prefix": model_prefix,
            "vocab_size": vocab_size,
            "character_coverage": character_coverage,
            "model_type": model_type,
            "input_sentence_size": 1000000,
            "shuffle_input_sentence": True,
            "pad_id": 0,
            "unk_id": 1,
            "bos_id": 2,
            "eos_id": 3,
        }
        
        if user_defined_symbols:
            train_args["user_defined_symbols"] = user_defined_symbols
            
        spm.SentencePieceTrainer.Train(" ".join([f"--{k}={v}" for k, v in train_args.items()]))
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size"""
        return self.sp.GetPieceSize()

def get_encoding(name: str) -> SPTokenizer:
    """Drop-in replacement for tiktoken.get_encoding()"""
    if os.path.exists(name):
        return SPTokenizer(name)
    return SPTokenizer.from_pretrained(name) 
