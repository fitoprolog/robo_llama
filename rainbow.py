import json
import random
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ColorSample:
    rgb: Tuple[float, float, float]  # RGB values between 0 and 1
    description: str

def generate_color_descriptions(rgb: Tuple[float, float, float]) -> List[str]:
    """Generate multiple descriptions for a color"""
    r, g, b = rgb
    
    # Color name templates
    templates = [
        "This is a {color} color with RGB values ({r:.2f}, {g:.2f}, {b:.2f}).",
        "The color shown is {color}, with red={r:.2f}, green={g:.2f}, and blue={b:.2f}.",
        "A {color} shade with RGB coordinates ({r:.2f}, {g:.2f}, {b:.2f}).",
        "This {color} hue has RGB components of {r:.2f}, {g:.2f}, and {b:.2f}.",
        "The displayed color is {color}, specified by RGB({r:.2f}, {g:.2f}, {b:.2f})."
    ]
    
    # Color descriptions based on RGB values
    if r > 0.8 and g < 0.2 and b < 0.2:
        color_name = "bright red"
    elif r < 0.2 and g > 0.8 and b < 0.2:
        color_name = "bright green"
    elif r < 0.2 and g < 0.2 and b > 0.8:
        color_name = "bright blue"
    elif r > 0.8 and g > 0.8 and b < 0.2:
        color_name = "yellow"
    elif r > 0.8 and g < 0.2 and b > 0.8:
        color_name = "magenta"
    elif r < 0.2 and g > 0.8 and b > 0.8:
        color_name = "cyan"
    elif r > 0.8 and g > 0.8 and b > 0.8:
        color_name = "white"
    elif r < 0.2 and g < 0.2 and b < 0.2:
        color_name = "black"
    elif r > 0.6 and g > 0.6 and b > 0.6:
        color_name = "light gray"
    elif r < 0.4 and g < 0.4 and b < 0.4:
        color_name = "dark gray"
    else:
        # For mixed colors, describe the dominant components
        components = []
        if r > 0.6:
            components.append("reddish")
        if g > 0.6:
            components.append("greenish")
        if b > 0.6:
            components.append("bluish")
        color_name = "-".join(components) if components else "mixed"
    
    # Generate descriptions using templates
    descriptions = []
    for template in templates:
        description = template.format(
            color=color_name,
            r=r,
            g=g,
            b=b
        )
        descriptions.append(description)
    
    return descriptions

def generate_dataset(num_samples: int = 1000) -> List[ColorSample]:
    """Generate a dataset of color samples with descriptions"""
    samples = []
    
    # Generate pure colors
    pure_colors = [
        (1.0, 0.0, 0.0),    # Red
        (0.0, 1.0, 0.0),    # Green
        (0.0, 0.0, 1.0),    # Blue
        (1.0, 1.0, 0.0),    # Yellow
        (1.0, 0.0, 1.0),    # Magenta
        (0.0, 1.0, 1.0),    # Cyan
        (1.0, 1.0, 1.0),    # White
        (0.0, 0.0, 0.0),    # Black
        (0.5, 0.5, 0.5),    # Gray
    ]
    
    # Add pure colors to dataset
    for rgb in pure_colors:
        descriptions = generate_color_descriptions(rgb)
        for description in descriptions:
            samples.append(ColorSample(rgb=rgb, description=description))
    
    # Generate random colors
    while len(samples) < num_samples:
        rgb = (
            random.random(),  # Random float between 0 and 1
            random.random(),
            random.random()
        )
        descriptions = generate_color_descriptions(rgb)
        for description in descriptions:
            samples.append(ColorSample(rgb=rgb, description=description))
    
    return samples

def save_dataset(samples: List[ColorSample], output_dir: str):
    """Save the dataset to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save text data
    text_data = []
    all_text = ""
    for sample in samples:
        all_text+= sample.description
        text_data.append({
            "text": sample.description,
            "rgb": sample.rgb
        })
 
    with open(output_dir / "raw_text.txt","w") as f:
        f.write(all_text)
    
    with open(output_dir / "text_data.jsonl", "w") as f:
        for item in text_data:
            f.write(json.dumps(item) + "\n")
    
    # Save RGB values as condition vectors
    conditions = np.array([sample.rgb for sample in samples], dtype=np.float32)
    np.save(output_dir / "conditions.npy", conditions)
    
    # Save metadata
    metadata = {
        "num_samples": len(samples),
        "condition_dim": 3,  # RGB values
        "description": "Color dataset with RGB values and descriptions"
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def main():
    # Generate dataset
    print("Generating color dataset...")
    samples = generate_dataset(num_samples=10000)
    
    # Save dataset
    print("Saving dataset...")
    save_dataset(samples, "color_dataset")
    
    print(f"Generated {len(samples)} samples")
    print("Dataset saved to 'color_dataset' directory")

if __name__ == "__main__":
    main()
