from bert_score import score as bert_score
from utils import load_file, init_csv, write_partial_results, REFERENCES_FILE, CANDIDATES_FILE

references = load_file(REFERENCES_FILE)
candidates = load_file(CANDIDATES_FILE)

init_csv()

P, R, F1 = bert_score(candidates, references, lang="es")
write_partial_results("BERTScore_F1", F1.tolist())
print("BERTScore calculado y guardado.")