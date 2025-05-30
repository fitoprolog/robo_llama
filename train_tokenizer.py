import json
import sentencepiece as spm
from pathlib import Path

def train_tokenizer():
    # Write texts to temporary file
    temp_file = 'whole_dataset.txt'
    
    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix="robollama",
        vocab_size=16000,
        model_type='bpe',
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        user_defined_symbols=['<mask>']
    )
    
if __name__ == '__main__':
    train_tokenizer() 
