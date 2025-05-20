import json
import sentencepiece as spm
from pathlib import Path

def train_tokenizer(input_file, model_prefix, vocab_size=2000):
    # Read all text data
    texts = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
    
    # Write texts to temporary file
    temp_file = 'temp_texts.txt'
    with open(temp_file, 'w') as f:
        for text in texts:
            f.write(text + '\n')
    
    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
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
    
    # Clean up temporary file
    Path(temp_file).unlink()

if __name__ == '__main__':
    train_tokenizer(
        input_file='color_dataset/text_data.jsonl',
        model_prefix='color_tokenizer',
        vocab_size=1000
    ) 
