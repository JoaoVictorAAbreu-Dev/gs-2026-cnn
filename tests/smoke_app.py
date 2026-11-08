"""Teste de fumaça: carrega o modelo salvo pelo treino e classifica imagens reais pelo app Gradio."""
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import app  # noqa: E402  (carrega outputs/best.keras e monta a interface sem iniciar o servidor)

split = json.loads((ROOT / "outputs" / "split.json").read_text())
acertos = 0
amostra = split["test"][:20]
for caminho in amostra:
    classe_real = Path(caminho).parent.name
    scores = app.classify(Image.open(ROOT / "data" / caminho))
    assert set(scores) == set(app.classes), scores
    assert abs(sum(scores.values()) - 1) < 1e-3, scores
    acertos += max(scores, key=scores.get) == classe_real
print(f"App respondeu para {len(amostra)} imagens de teste; acertos: {acertos}/{len(amostra)}")
assert acertos >= 12, "Acurácia muito abaixo do esperado no teste de fumaça"
