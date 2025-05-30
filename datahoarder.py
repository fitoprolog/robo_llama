import datasets
import json 
import re
import struct
import numpy as np
import splitter

def replace_numbers_with_hex32(text):
    pattern = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')

    def replacer(match):
        num_str = match.group()
        try:
            if '.' in num_str:
                num = float(num_str)
                hex32 = hex(struct.unpack('<I', struct.pack('<f', num))[0])
            else:
                num = int(num_str)
                hex32 = hex(np.int32(num).view(np.uint32))
        except:
            return "OVERFLOW"
        return hex32

    return pattern.sub(replacer, text)

ALL_DATASET=[]

def prepare_datasets():
    math_expr_dataset = datasets.load_dataset("Gusarich/math-expressions-1m")["train"]
    nl_2_cypher_dataset = datasets.load_dataset("megagonlabs/cypherbench")["train"]
    pythonic_function_calling_dataset = datasets.load_dataset("driaforall/pythonic-function-calling")["train"]
    urban_dictionary_dataset = datasets.load_dataset("georgiyozhegov/urbandictionary-raw")["train"]
    emojis_dataset = datasets.load_dataset("badrex/LLM-generated-emoji-descriptions")["train"]
    code_dataset = datasets.load_dataset("open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled")["train"]

    with open("whole_dataset.txt", "w") as fd:
        """
        for sample in code_dataset:
            line =f'CODE:{sample["gold_standard_solution"]}\n'
            ALL_DATASET.append(line)
            fd.write(line)
        """
        for sample in emojis_dataset:
            line = f'EMOJI:{sample["unicode"].lower()}\n:{sample["LLM description"]}\n'
            line = replace_numbers_with_hex32(line)
            fd.write(line)
            ALL_DATASET.append(line)
        
        with open("local_data/dictionary.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
        parsed_data = splitter.process_dictionary(raw_text)
                
        for sample in parsed_data:
            line = f'NORMAL_DICTIONARY:{sample["word"].lower()}\nDEFINITION:{sample["definition"]}\n'
            line = replace_numbers_with_hex32(line)
            fd.write(line)
            ALL_DATASET.append(line)

        for sample in math_expr_dataset:
            if sample["result"] == "INVALID":
                continue
            line = f'MATH:{sample["expression"]} RESULT:{sample["result"]}\n'
            line = replace_numbers_with_hex32(line)
            fd.write(line)
            ALL_DATASET.append(line)

        for sample in nl_2_cypher_dataset:
            line = f'CYPHER_QUESTION:{sample["nl_question"]}\nCYPHER:{sample["gold_cypher"]}\n'
            fd.write(replace_numbers_with_hex32(line))
            ALL_DATASET.append(line)

        for sample in pythonic_function_calling_dataset:
            line = f'PYTHON_SIGNATURE:{sample["tools"]}\n'
            fd.write(line)
            ALL_DATASET.append(line)

        for sample in urban_dictionary_dataset:
            line = f'URBAN_DICTIONARY:{sample["word"]}\nDEFINITION:{sample["definition"]}\n'
            fd.write(line)
            ALL_DATASET.append(line)

        fd.close()
        with open("all_dataset.json","w") as fd:
            fd.write(json.dumps(ALL_DATASET))

if __name__ == "__main__":
    prepare_datasets()
