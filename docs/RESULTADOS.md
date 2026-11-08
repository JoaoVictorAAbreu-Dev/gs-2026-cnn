# Resultados dos experimentos

Execução registrada em 05/11/2026 — EuroSAT RGB oficial (Zenodo 7711810, SHA-256 do arquivo `b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90`), 800 imagens, divisão estratificada 480/160/160, semente 42, CPU, TensorFlow 2.20, Python 3.12. Dados completos em `outputs/results.json`.

| Modelo | Acurácia validação | F1 macro validação | Menor perda validação | Épocas | Parâmetros |
|---|---:|---:|---:|---:|---:|
| CNN 1 | 0,8500 | 0,849 | 0,437 | 12 | 529.700 |
| **CNN 2** | **0,8688** | **0,867** | **0,331** | 12 | 2.117.188 |
| CNN 3 | 0,8313 | 0,829 | 0,451 | 10 | 1.142.468 |

## Teste (CNN 2)

Acurácia 0,850 · precisão macro 0,850 · recall macro 0,850 · F1 macro 0,849.

| Real \ Previsto | Forest | Industrial | Residential | River |
|---|---:|---:|---:|---:|
| Forest | **37** | 0 | 0 | 3 |
| Industrial | 0 | **32** | 8 | 0 |
| Residential | 0 | 6 | **29** | 5 |
| River | 2 | 0 | 0 | **38** |

## Análise

- **CNN 1 → CNN 2:** mais filtros (32/64), densa maior e dropout 0,25 aumentaram a acurácia de validação em 1,9 p.p. e reduziram a perda em 24 %.
- **CNN 2 → CNN 3:** um terceiro bloco com dropout 0,35 e taxa 0,0005 resultou em subajuste dentro de 12 épocas (treino 76,7 % × validação 83,1 % na melhor época).
- **Sobreajuste:** a CNN 1, sem dropout, terminou com 91,9 % no treino e 85,0 % na validação; a CNN 2 reduziu o gap para cerca de 4 p.p.
- **Erros:** 14 dos 24 erros de teste estão entre Industrial e Residential; Residential também foi confundida com River (5 casos).
- **Validação × teste:** 86,9 % × 85,0 % — diferença pequena, coerente com uma seleção sem vazamento.
- Como cada experimento alterou mais de um hiperparâmetro, a comparação mostra o efeito do conjunto de mudanças.

## Reprodutibilidade e aplicação

O notebook foi executado integralmente em diretório separado e reproduziu a seleção e a matriz de teste. A aplicação Gradio foi iniciada localmente e uma imagem real de Forest enviada pela interface retornou as quatro pontuações, com Forest como classe mais provável.
