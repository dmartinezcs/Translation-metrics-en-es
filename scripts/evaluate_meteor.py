import nltk
from nltk.translate.meteor_score import meteor_score
from utils import load_file, init_csv, write_partial_results, REFERENCES_FILE, CANDIDATES_FILE

nltk.download('wordnet')
references = load_file(REFERENCES_FILE)
candidates = load_file(CANDIDATES_FILE)

init_csv()

meteor_scores = []
for ref, cand in zip(references, candidates):
    meteor_scores.append(meteor_score([ref.split()], cand.split()))

write_partial_results("METEOR", meteor_scores)
print("METEOR calculado y guardado.")