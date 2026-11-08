# Resultados dos experimentos

Execução registrada: notebook executado do início ao fim no GitHub Actions (Ubuntu, CPU, Python 3.12, TensorFlow 2.20) — EuroSAT RGB oficial (Zenodo 7711810), 800 imagens, divisão estratificada 480/160/160, semente 42, treino determinístico. Dados completos em `outputs/results.json`.

| Modelo | Mudança | Acurácia val. | F1 macro val. | Menor perda val. | Épocas (melhor) | Treino na melhor época | Parâmetros |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN 1 | base (16/32, Dense 64) | 0,8438 | 0,843 | 0,448 | 12 (12) | 0,931 | 529.700 |
| **CNN 2** | 32/64, Dense 128, Dropout 0,25 | **0,8875** | **0,886** | 0,335 | 12 (10) | 0,954 | 2.117.188 |
| CNN 3 | 32/64/128, Dropout 0,35, LR 0,0005 | 0,8250 | 0,819 | 0,416 | 12 (12) | 0,840 | 1.142.468 |
| CNN 4 | CNN 2 + BatchNorm (momentum 0,9) | 0,8375 | 0,837 | 0,356 | 10 (7) | 0,850 | 2.117.476 |
| CNN 5 | CNN 2 + aumento de dados | 0,8688 | 0,869 | 0,397 | 12 (12) | 0,802 | 2.117.188 |
| CNN 6 | CNN 2 + até 30 épocas | 0,8813 | 0,881 | **0,331** | 20 (15) | 0,962 | 2.117.188 |

Seleção (definida antes dos testes): maior acurácia de validação; desempate pela menor perda → **CNN 2**.

## Teste (CNN 2)

Acurácia 0,856 · precisão macro 0,856 · recall macro 0,856 · F1 macro 0,856.

| Real \ Previsto | Forest | Industrial | Residential | River |
|---|---:|---:|---:|---:|
| Forest | **37** | 0 | 0 | 3 |
| Industrial | 0 | **33** | 7 | 0 |
| Residential | 0 | 7 | **30** | 3 |
| River | 2 | 0 | 1 | **37** |

## Análise

- **Primeira rodada:** mais filtros, densa maior e dropout (CNN 2) foram a melhor combinação; a rede mais profunda e mais regularizada (CNN 3) teve subajuste.
- **BatchNorm (CNN 4):** perda de validação instável no início (4,28 na 1ª época) e oscilante; sem ganho em uma rede rasa com 480 imagens.
- **Aumento de dados (CNN 5):** treino abaixo da validação na melhor época — sobreajuste eliminado; a perda ainda caía na 12ª época.
- **Mais épocas (CNN 6):** as 12 primeiras épocas reproduzem exatamente a CNN 2 (treino determinístico); a perda melhorou até a 15ª época e depois subiu com o treino acima de 96 % — início de sobreajuste.
- **Erros:** 14 dos 23 erros de teste estão entre Industrial e Residential.
- **Validação × teste:** 88,8 % × 85,6 % — diferença pequena, coerente com uma seleção sem vazamento.

## Aplicação

O teste automático (`tests/smoke_app.py`) carrega o modelo selecionado e classifica 20 imagens de teste pela mesma função do app Gradio: 19 acertos, com as quatro probabilidades somando 1 em todos os casos. O notebook mostra o app classificando uma imagem de cada classe.
