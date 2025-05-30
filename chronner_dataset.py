import random
import json
import re
from cron_descriptor import get_description

# 🔧 Configuraciones de distorsión
def random_remove_commas(text):
    if random.random() < 0.5:
        return text.replace(",", " ")  # ahora reemplaza por espacio
    return text

def random_remove_spaces(text):
    if random.random() < 0.3:
        return text.replace(" ", "")
    return text

def random_capitalize(text):
    choice = random.choice(['lower', 'upper', 'capitalize'])
    if choice == 'lower':
        return text.lower()
    elif choice == 'upper':
        return text.upper()
    else:
        return text.capitalize()

def random_alter_terms(text):
    replacements = {
        "only": random.choice(["just", "", ""]),
        "on the": random.choice(["during", "at", ""]),
        "past the hour": random.choice(["after the hour", "past"]),
        "at": random.choice(["@", "", ""]),
        "every": random.choice(["each", "every", ""]),
        "minute past": random.choice(["after minute", "with minutes", "and minutes", ""],),
        "minute": random.choice(["min", "minutes", "mins"]),
        "hour": random.choice(["hour", "hr", "hrs"]),
        "hour {0}".format(random.randint(1,12)): random.choice(["{0} hour".format(random.randint(1,12)), "hour {0} with minutes".format(random.randint(1,12)), "{0} and minutes".format(random.randint(1,12))]),
    }
    for key, val in replacements.items():
        text = text.replace(key, val)
    return text

def remove_leading_zeros(text):
    # remueve ceros a la izquierda y detecta horas tipo 06:42 para convertir en 6:42 o 6 42
    text = re.sub(r'\b0+(\d)', r'\1', text)
    if random.random() < 0.5:
        text = re.sub(r'(\d{1,2}):(\d{2})', lambda m: f"{int(m.group(1))} {int(m.group(2))}", text)
    return text

def permutar_frases(texto):
    partes = texto.split(", ")
    random.shuffle(partes)
    texto = ", ".join(partes)
    texto = random_remove_commas(texto)
    texto = remove_leading_zeros(texto)
    texto = random_alter_terms(texto)
    texto = random_capitalize(texto)
    texto = random_remove_spaces(texto)
    return texto

def generar_cron_aleatorio_permutado(locale='en'):
    minuto = random.choice(['*', str(random.randint(0, 59)), f"*/{random.randint(1, 30)}"])
    hora = random.choice(['*', str(random.randint(0, 23)), f"*/{random.randint(1, 12)}"])
    dia_mes = random.choice(['*', str(random.randint(1, 28))])
    mes = random.choice(['*', str(random.randint(1, 12))])
    dia_semana = random.choice(['*', str(random.randint(0, 6))])

    expr = f"{minuto} {hora} {dia_mes} {mes} {dia_semana}"

    try:
        descripcion = get_description(expr)
        descripcion_permutada = permutar_frases(descripcion)
    except Exception as e:
        descripcion_permutada = f"Error: {str(e)}"

    return f"NL: {descripcion_permutada} CRON: {expr} XXX"

# 🔪 Generar dataset
resultados = []
for _ in range(int(10e6)):
    resultado = generar_cron_aleatorio_permutado()
    resultados.append(resultado)

# 📂 Guardar en JSON como lista de strings
with open("cron_dataset.json", "w") as f:
    json.dump(resultados, f, indent=2)

print("✅ Dataset generado y guardado como cron_dataset.json")

