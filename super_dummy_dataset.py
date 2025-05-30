import random
import json

# Mapping A-D → W-Z (casing preserved)
pair_map = {
    'A': 'W', 'B': 'X', 'C': 'Y', 'D': 'Z',
    'a': 'w', 'b': 'x', 'c': 'y', 'd': 'z'
}

def generate_sequence(max_total_length=512):
    input_length = random.randint(1, max_total_length // 2)
    input_seq = ''.join(random.choice(['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd']) for _ in range(input_length))
    continuation = ''.join(pair_map[ch] for ch in input_seq)
    return input_seq + '=' + continuation  # 🔥 insert inference marker

def generate_dataset(n_sequences=1000000, output_file="dummy_dataset.json"):
    dataset = [generate_sequence() for _ in range(n_sequences)]
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ Dataset with '=' markers saved to {output_file}")

# Run it 😈
generate_dataset()

