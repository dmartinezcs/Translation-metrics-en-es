import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
RESULTS_DIR = os.path.join(BASE_DIR, "../results")

SOURCE_FILE = os.path.join(DATA_DIR, "source.txt")
REFERENCES_FILE = os.path.join(DATA_DIR, "references.txt")
CANDIDATES_FILE = os.path.join(DATA_DIR, "candidates.txt")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "evaluation.csv")

# Crear carpeta results si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def init_csv():
    """Inicializa el CSV con cabecera vacía o si no existe"""
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Source (EN)", "Reference (ES)", "Candidate (ES)",
                             "BLEU", "METEOR", "BERTScore_F1", "COMET"])

def write_partial_results(metric_name, scores):
    """Escribe o actualiza la columna de una métrica en el CSV"""
    # Leer CSV actual
    rows = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

    # Si solo está la cabecera, inicializamos con datos base
    if len(rows) <= 1:
        sources = load_file(SOURCE_FILE)
        references = load_file(REFERENCES_FILE)
        candidates = load_file(CANDIDATES_FILE)
        rows = [["Source (EN)", "Reference (ES)", "Candidate (ES)",
                 "BLEU", "METEOR", "BERTScore_F1", "COMET"]]
        for s, r, c in zip(sources, references, candidates):
            rows.append([s, r, c, "", "", "", ""])

    # Determinar columna
    header = rows[0]
    if metric_name not in header:
        raise ValueError(f"{metric_name} no está en el CSV")
    col_idx = header.index(metric_name)

    # Escribir los valores
    for i, score in enumerate(scores):
        rows[i+1][col_idx] = round(score, 4) if isinstance(score, float) else score

    # Guardar CSV actualizado
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)