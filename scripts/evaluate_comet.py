from comet.models import download_model, load_from_checkpoint
from utils import load_file, init_csv, write_partial_results, REFERENCES_FILE, CANDIDATES_FILE, SOURCE_FILE

sources = load_file(SOURCE_FILE)
references = load_file(REFERENCES_FILE)
candidates = load_file(CANDIDATES_FILE)

init_csv()

print("Descargando y cargando modelo COMET...")
model_path = download_model("unbabel/wmt20-comet-da")  # modelo más ligero
print("Model path")
comet_model = load_from_checkpoint(model_path)
print("comet model")
print("Modelo COMET cargado.")

comet_scores = []
batch_size = 4  #Test how many batches can codespace handle
for i in range(0, len(sources), batch_size):
    batch = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(
            sources[i:i+batch_size],
            candidates[i:i+batch_size],
            references[i:i+batch_size]
        )
    ]

    batch_output = comet_model.predict(batch, batch_size=batch_size)
    print(type(batch_output["scores"][0]), batch_output["scores"][0])
    comet_scores.extend(batch_output["scores"])

write_partial_results("COMET", comet_scores)
print("COMET calculado y guardado.")

comet_mean = sum(comet_scores) / len(comet_scores)

print(f"COMET medio: {comet_mean:.4f}")
print("COMET calculado y guardado.")