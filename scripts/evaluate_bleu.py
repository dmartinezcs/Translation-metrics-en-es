from nltk.translate.bleu_score import sentence_bleu
from utils import load_file, init_csv, write_partial_results, REFERENCES_FILE, CANDIDATES_FILE, SOURCE_FILE

sources = load_file(SOURCE_FILE)
references = load_file(REFERENCES_FILE)
candidates = load_file(CANDIDATES_FILE)
init_csv()

bleu_scores = []
for ref, cand in zip(references, candidates):
    ref_tokens = [ref.split()]
    cand_tokens = cand.split()
    bleu_scores.append(sentence_bleu(ref_tokens, cand_tokens))

write_partial_results("BLEU", bleu_scores)
print("BLEU calculado y guardado.")