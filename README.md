# Multimodal LLaMA with SentencePiece Tokenizer

This is a modified version of Karpathy's LLaMA implementation that supports multimodal inputs and uses SentencePiece tokenizer instead of tiktoken.

## Changes from Original

- Replaced tiktoken with SentencePiece for tokenization
- Added support for multimodal inputs through conditioning vectors
- Added training scripts for custom SentencePiece models
- Improved data loading and preprocessing pipeline

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training a SentencePiece Model

Before training the model, you need to train a SentencePiece tokenizer on your data:

```bash
python train_tokenizer.py \
    --input your_text_file.txt \
    --model_prefix sp_model \
    --vocab_size 32000 \
    --character_coverage 0.9995 \
    --model_type bpe
```

This will create two files:
- `sp_model.model`: The trained SentencePiece model
- `sp_model.vocab`: The vocabulary file (human readable)

## Preprocessing Data

After training the tokenizer, preprocess your text data:

```bash
python train.py \
    --preprocess \
    --raw_text_file your_text_file.txt \
    --text_path processed_data.jsonl \
    --tokenizer_name sp_model
```

## Training the Model

Train the model with your preprocessed data:

```bash
python train.py \
    --text_path processed_data.jsonl \
    --condition_path condition_vectors.npy \
    --out_dir checkpoints \
    --tokenizer_name sp_model \
    --dim 4096 \
    --n_layers 32 \
    --n_heads 32 \
    --vocab_size 32000 \
    --max_seq_len 2048 \
    --batch_size 12 \
    --learning_rate 6e-4 \
    --max_iters 20000
```

## Generating Text

Generate text using the trained model:

```bash
python inference_cli.py \
    --checkpoint checkpoints/checkpoint.pt \
    --tokenizer_name sp_model \
    --temperature 0.8 \
    --max_new_tokens 200
```

## Model Architecture

The model architecture remains the same as the original LLaMA implementation, with the addition of:
- Conditioning vector input for multimodal support
- SentencePiece tokenizer integration
- Modified data loading pipeline

## File Structure

- `tokenizer.py`: SentencePiece tokenizer implementation
- `train_tokenizer.py`: Script for training SentencePiece models
- `model.py`: Model architecture implementation
- `dataloader.py`: Data loading and preprocessing utilities
- `train.py`: Training script
- `inference.py`: Inference utilities
- `inference_cli.py`: Command-line interface for text generation

## Notes

1. The SentencePiece tokenizer is more flexible than tiktoken and allows training custom vocabularies.
2. Make sure to train the tokenizer on a representative sample of your text data.
3. The model expects the same vocabulary size as specified during tokenizer training.
4. When using a pretrained model, make sure to use the same SentencePiece model that was used during training. 