import random
import json
import re
from cron_descriptor import get_description

# 🔧 Enhanced distortion configurations with much more entropy

def add_typos_and_variations(text):
    """Add common typos and character variations"""
    typo_map = {
        'e': ['3', 'e'],
        'o': ['0', 'o'],
        'a': ['@', 'a'],
        's': ['$', '5', 's'],
        'i': ['1', '!', 'i'],
        'l': ['1', '|', 'l'],
        'every': ['evry', 'every', 'evrry'],
        'minute': ['minite', 'min', 'minutes'],
        'second': ['sec', 'seconds', 'secs'],
        'hour': ['hr', 'hrs', 'hour', 'hours'],
        'day': ['dy', 'days', 'day'],
        'week': ['wk', 'weeks', 'week'],
        'month': ['mth', 'months', 'month'],
        'year': ['yr', 'years', 'year']
    }
    
    if random.random() < 0.3:  # 30% chance of typos
        for original, variants in typo_map.items():
            if original in text.lower():
                text = text.replace(original, random.choice(variants))
    
    return text

def add_informal_language(text):
    """Add slang and informal expressions"""
    informal_replacements = {
        "every": random.choice(["every", "each", "all", "per"]),
        "at": random.choice(["at", "@", "by", "around", "near"]),
        "on": random.choice(["on", "during", "in", "at"]),
        "minute": random.choice(["minute", "min", "mins", "m"]),
        "hour": random.choice(["hour", "hr", "hrs", "h", "o'clock"]),
        "day": random.choice(["day", "days", "daily"]),
        "week": random.choice(["week", "weeks", "weekly", "wk"]),
        "month": random.choice(["month", "months", "monthly", "mth"]),
        "morning": random.choice(["morning", "AM", "a.m.", "am"]),
        "afternoon": random.choice(["afternoon", "PM", "p.m.", "pm"]),
        "evening": random.choice(["evening", "night", "PM", "p.m."]),
        "midnight": random.choice(["midnight", "12am", "00:00", "12:00am"]),
        "noon": random.choice(["noon", "12pm", "12:00", "12:00pm"])
    }
    
    for original, replacement in informal_replacements.items():
        if original in text.lower():
            text = text.replace(original, replacement)
    
    return text

def add_time_format_variations(text):
    """Add different time format representations"""
    # Convert 24h to 12h and vice versa
    time_patterns = [
        (r'(\d{1,2}):(\d{2})', lambda m: f"{m.group(1)}:{m.group(2)}"),  # Keep as is
        (r'(\d{1,2}):(\d{2})', lambda m: f"{m.group(1)} {m.group(2)}"),  # Remove colon
        (r'(\d{1,2}):(\d{2})', lambda m: f"{m.group(1)}.{m.group(2)}"),  # Dot separator
        (r'(\d{1,2}):(\d{2})', lambda m: f"{m.group(1)}-{m.group(2)}")   # Dash separator
    ]
    
    if random.random() < 0.4:
        pattern, replacement = random.choice(time_patterns)
        text = re.sub(pattern, replacement, text)
    
    return text

def add_natural_language_variations(text):
    """Add more natural language patterns"""
    variations = {
        "every day": random.choice(["daily", "each day", "every single day", "all days"]),
        "every week": random.choice(["weekly", "each week", "every single week", "all weeks"]),
        "every month": random.choice(["monthly", "each month", "every single month", "all months"]),
        "every year": random.choice(["yearly", "annually", "each year", "every single year"]),
        "monday": random.choice(["monday", "mon", "mondays", "every monday"]),
        "tuesday": random.choice(["tuesday", "tue", "tues", "tuesdays"]),
        "wednesday": random.choice(["wednesday", "wed", "wednesdays"]),
        "thursday": random.choice(["thursday", "thu", "thurs", "thursdays"]),
        "friday": random.choice(["friday", "fri", "fridays"]),
        "saturday": random.choice(["saturday", "sat", "saturdays"]),
        "sunday": random.choice(["sunday", "sun", "sundays"]),
        "january": random.choice(["january", "jan", "1st month"]),
        "february": random.choice(["february", "feb", "2nd month"]),
        "march": random.choice(["march", "mar", "3rd month"]),
        "april": random.choice(["april", "apr", "4th month"]),
        "may": random.choice(["may", "5th month"]),
        "june": random.choice(["june", "jun", "6th month"]),
        "july": random.choice(["july", "jul", "7th month"]),
        "august": random.choice(["august", "aug", "8th month"]),
        "september": random.choice(["september", "sep", "sept", "9th month"]),
        "october": random.choice(["october", "oct", "10th month"]),
        "november": random.choice(["november", "nov", "11th month"]),
        "december": random.choice(["december", "dec", "12th month"])
    }
    
    text_lower = text.lower()
    for original, replacement in variations.items():
        if original in text_lower:
            text = text.replace(original, replacement)
    
    return text

def add_punctuation_variations(text):
    """Add or remove punctuation randomly"""
    if random.random() < 0.3:
        # Remove punctuation
        text = re.sub(r'[.,;:!?]', '', text)
    elif random.random() < 0.2:
        # Add random punctuation
        punctuation = random.choice(['.', ',', ';', '!'])
        words = text.split()
        if len(words) > 1:
            insert_pos = random.randint(1, len(words) - 1)
            words[insert_pos] = words[insert_pos] + punctuation
            text = ' '.join(words)
    
    return text

def add_numeric_variations(text):
    """Convert between numeric and word representations"""
    number_words = {
        '1': random.choice(['1', 'one', 'first']),
        '2': random.choice(['2', 'two', 'second']),
        '3': random.choice(['3', 'three', 'third']),
        '4': random.choice(['4', 'four', 'fourth']),
        '5': random.choice(['5', 'five', 'fifth']),
        '6': random.choice(['6', 'six', 'sixth']),
        '7': random.choice(['7', 'seven', 'seventh']),
        '8': random.choice(['8', 'eight', 'eighth']),
        '9': random.choice(['9', 'nine', 'ninth']),
        '10': random.choice(['10', 'ten', 'tenth']),
        '11': random.choice(['11', 'eleven', 'eleventh']),
        '12': random.choice(['12', 'twelve', 'twelfth'])
    }
    
    for num, word_variant in number_words.items():
        if num in text:
            text = text.replace(num, word_variant)
    
    return text

def add_contextual_words(text):
    """Add contextual words that don't change meaning but add variety"""
    prefixes = ["", "please ", "I need ", "schedule ", "run ", "execute "]
    suffixes = ["", " please", " thanks", " every time", " always", " consistently"]
    
    if random.random() < 0.3:
        text = random.choice(prefixes) + text
    if random.random() < 0.3:
        text = text + random.choice(suffixes)
    
    return text

def add_whitespace_variations(text):
    """Add irregular whitespace patterns"""
    if random.random() < 0.4:
        # Add extra spaces randomly
        words = text.split()
        result = []
        for word in words:
            result.append(word)
            if random.random() < 0.3:
                result.append(' ' * random.randint(1, 3))
        text = ' '.join(result)
    
    return text

def add_abbreviation_variations(text):
    """Add common abbreviations and expansions"""
    abbrev_map = {
        'am': random.choice(['am', 'a.m.', 'AM', 'A.M.']),
        'pm': random.choice(['pm', 'p.m.', 'PM', 'P.M.']),
        'mon': random.choice(['mon', 'monday', 'Mon', 'Monday']),
        'tue': random.choice(['tue', 'tuesday', 'Tue', 'Tuesday']),
        'wed': random.choice(['wed', 'wednesday', 'Wed', 'Wednesday']),
        'thu': random.choice(['thu', 'thursday', 'Thu', 'Thursday']),
        'fri': random.choice(['fri', 'friday', 'Fri', 'Friday']),
        'sat': random.choice(['sat', 'saturday', 'Sat', 'Saturday']),
        'sun': random.choice(['sun', 'sunday', 'Sun', 'Sunday'])
    }
    
    for abbrev, full_form in abbrev_map.items():
        if abbrev in text.lower():
            text = text.replace(abbrev, full_form)
    
    return text

# Enhanced distortion functions
def random_remove_commas(text):
    if random.random() < 0.5:
        return text.replace(",", random.choice([" ", "", " and "]))
    return text

def random_remove_spaces(text):
    if random.random() < 0.3:
        return text.replace(" ", "")
    return text

def random_capitalize(text):
    choice = random.choice(['lower', 'upper', 'capitalize', 'title', 'random'])
    if choice == 'lower':
        return text.lower()
    elif choice == 'upper':
        return text.upper()
    elif choice == 'title':
        return text.title()
    elif choice == 'random':
        return ''.join(c.upper() if random.random() < 0.5 else c.lower() for c in text)
    else:
        return text.capitalize()

def random_alter_terms(text):
    replacements = {
        "only": random.choice(["just", "solely", "exclusively", ""]),
        "on the": random.choice(["during the", "at the", "in the", ""]),
        "past the hour": random.choice(["after the hour", "past", "minutes past"]),
        "at": random.choice(["@", "by", "around", "near", ""]),
        "every": random.choice(["each", "all", "per", "every single", ""]),
        "minute past": random.choice(["after minute", "with minutes", "and minutes", "min past"]),
        "minute": random.choice(["min", "minutes", "mins", "m"]),
        "hour": random.choice(["hour", "hr", "hrs", "h"]),
        "between": random.choice(["from", "starting at", "in the range"]),
        "and": random.choice(["and", "&", "to", "through"]),
        "of": random.choice(["of", "in", "during"]),
        "the": random.choice(["the", "", "each"])
    }
    
    for key, val in replacements.items():
        if key in text.lower():
            text = text.replace(key, val)
    
    return text

def remove_leading_zeros(text):
    text = re.sub(r'\b0+(\d)', r'\1', text)
    if random.random() < 0.5:
        text = re.sub(r'(\d{1,2}):(\d{2})', 
                     lambda m: f"{int(m.group(1))} {int(m.group(2))}", text)
    return text

def permutar_frases(texto):
    """Enhanced phrase permutation with all entropy functions"""
    # Apply all distortion functions in random order
    distortion_functions = [
        add_typos_and_variations,
        add_informal_language,
        add_time_format_variations,
        add_natural_language_variations,
        add_punctuation_variations,
        add_numeric_variations,
        add_contextual_words,
        add_whitespace_variations,
        add_abbreviation_variations,
        random_remove_commas,
        remove_leading_zeros,
        random_alter_terms,
        random_capitalize,
        random_remove_spaces
    ]
    
    # Randomly shuffle and apply 3-7 distortion functions
    random.shuffle(distortion_functions)
    num_distortions = random.randint(3, 7)
    
    for func in distortion_functions[:num_distortions]:
        texto = func(texto)
    
    # Additional sentence structure variations
    if random.random() < 0.3:
        partes = texto.split(", ")
        random.shuffle(partes)
        texto = ", ".join(partes)
    
    return texto

def generar_cron_aleatorio_permutado(locale='en'):
    """Generate more diverse cron expressions"""
    # More diverse cron pattern generation
    minuto_patterns = ['*', str(random.randint(0, 59)), 
                      f"*/{random.randint(1, 30)}", 
                      f"{random.randint(0, 30)}-{random.randint(31, 59)}",
                      f"{random.randint(0, 15)},{random.randint(30, 45)}"]
    
    hora_patterns = ['*', str(random.randint(0, 23)), 
                    f"*/{random.randint(1, 12)}", 
                    f"{random.randint(0, 11)}-{random.randint(12, 23)}",
                    f"{random.randint(0, 6)},{random.randint(12, 18)}"]
    
    dia_mes_patterns = ['*', str(random.randint(1, 28)), 
                       f"*/{random.randint(1, 15)}", 
                       f"{random.randint(1, 15)}-{random.randint(16, 28)}"]
    
    mes_patterns = ['*', str(random.randint(1, 12)), 
                   f"*/{random.randint(1, 6)}", 
                   f"{random.randint(1, 6)},{random.randint(7, 12)}"]
    
    dia_semana_patterns = ['*', str(random.randint(0, 6)), 
                          f"{random.randint(0, 2)},{random.randint(4, 6)}",
                          f"{random.randint(1, 5)}"]
    
    minuto = random.choice(minuto_patterns)
    hora = random.choice(hora_patterns)
    dia_mes = random.choice(dia_mes_patterns)
    mes = random.choice(mes_patterns)
    dia_semana = random.choice(dia_semana_patterns)
    
    expr = f"{minuto} {hora} {dia_mes} {mes} {dia_semana}"
    
    try:
        descripcion = get_description(expr)
        descripcion_permutada = permutar_frases(descripcion)
    except Exception as e:
        descripcion_permutada = f"Error: {str(e)}"
    
    return f"NL: {descripcion_permutada} CRON: {expr} XXX"

# Generate enhanced dataset
resultados = []
print("🚀 Generating enhanced dataset with maximum entropy...")

for i in range(int(1e6)):  # Start with 1M for testing
    if i % 100000 == 0:
        print(f"Generated {i} samples...")
    
    resultado = generar_cron_aleatorio_permutado()
    resultados.append(resultado)

# Save to JSON
with open("enhanced_cron_dataset.json", "w") as f:
    json.dump(resultados, f, indent=2)

print("✅ Enhanced dataset generated and saved as enhanced_cron_dataset.json")
print(f"📊 Total samples: {len(resultados)}")

# Show some samples
print("\n📋 Sample outputs:")
for i in range(5):
    print(f"{i+1}. {resultados[i]}")
