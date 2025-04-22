import numpy as np
import json
import random
import os

# Define rainbow-ish colors with RGB normalized (0 to 1)
COLOR_NAMES = [
    ("red", [1.0, 0.0, 0.0]),
    ("orange", [1.0, 0.5, 0.0]),
    ("yellow", [1.0, 1.0, 0.0]),
    ("green", [0.0, 1.0, 0.0]),
    ("cyan", [0.0, 1.0, 1.0]),
    ("blue", [0.0, 0.0, 1.0]),
    ("violet", [0.5, 0.0, 1.0]),
    ("magenta", [1.0, 0.0, 1.0]),
]

PHRASE_TEMPLATES = [
    "Look at this beautiful {} and {} blend.",
    "Here we see some {} mixed with {} tones.",
    "A dash of {} alongside a touch of {}.",
    "Such rich {} combined with subtle {}.",
    "Oh this is some {} and {} magic.",
    "Deep {} melts into fresh {}.",
]

def generate_dataset(
    output_jsonl: str,
    output_npy: str,
    num_samples: int = 100,
    condition_dim: int = 6,  # 3 + 3 (RGB + RGB)
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    # Filter out low intensity colors
    valid_colors = [c for c in COLOR_NAMES if max(c[1]) >= 0.3]
    
    texts = []
    conditions = []

    for _ in range(num_samples):
        # Pick two distinct colors with enough brightness
        color1, vec1 = random.choice(valid_colors)
        color2, vec2 = random.choice(valid_colors)
        while color2 == color1:
            color2, vec2 = random.choice(valid_colors)
        
        # Add fuzziness to each color
        fuzziness = np.random.uniform(-0.1, 0.1, size=3)
        vec1_fuzzy = np.clip(np.array(vec1) + fuzziness, 0, 1)
        vec2_fuzzy = np.clip(np.array(vec2) + fuzziness[::-1], 0, 1)

        # Combine to form condition vector
        condition = np.concatenate([vec1_fuzzy, vec2_fuzzy])
        conditions.append(condition)

        # Generate descriptive text
        template = random.choice(PHRASE_TEMPLATES)
        phrase = template.format(color1, color2)
        texts.append({"text": phrase})
    
    # Save text data
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in texts:
            f.write(json.dumps(item) + "\n")

    # Save condition vectors
    np.save(output_npy, np.array(conditions, dtype=np.float32))

    print(f"✅ Generated {num_samples} samples")
    print(f"📝 Texts saved to: {output_jsonl}")
    print(f"🧠 Conditions saved to: {output_npy}")

# Run directly
if __name__ == "__main__":
    os.makedirs("rainbow_dataset", exist_ok=True)
    generate_dataset(
        output_jsonl="rainbow_dataset/text.jsonl",
        output_npy="rainbow_dataset/conditions.npy",
        num_samples=200,
    )
