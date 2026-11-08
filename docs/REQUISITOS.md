# Critérios de avaliação × evidências

| Critério (PDF, seção 9) | Pontos | Evidência |
|---|---:|---|
| Preparação adequada dos dados | 1,0 | 800 imagens (200/classe), RGB 64 × 64, pixels em [0, 1], divisão estratificada 480/160/160 sem sobreposição (`split.json`, notebook §2) |
| Implementação e compreensão da CNN | 2,0 | Conv2D, MaxPooling2D, ReLU, Dense e softmax, sem pré-treino (`src/train.py`); papel de cada camada e contagem de parâmetros camada a camada conferida com o Keras (notebook §3) |
| Experimentação e otimização | 2,0 | 3 arquiteturas com hipóteses explícitas (filtros, profundidade, densa, dropout, taxa de aprendizado), seleção só pela validação (notebook §3–4, slides 4–5) |
| Avaliação e interpretação | 1,5 | Acurácia, precisão, recall, F1 por classe e macro, matriz de confusão absoluta e normalizada, curvas de perda/acurácia, análise de sobreajuste e subajuste (notebook §4–5, slides 6–8) |
| Aplicação web | 1,5 | Gradio com upload, classificação, exibição da imagem e pontuação das 4 classes; link público no Colab e execução local (`src/app.py`, notebook §6, slide 9) |
| Decisões e uso crítico de IA | 1,0 | Tabela de contribuições da IA e como cada uma foi avaliada; sugestão descartada com base nos resultados (notebook §7, slide 10) |
| Clareza e qualidade dos materiais | 1,0 | Notebook com seções numeradas, 10 slides, README com instruções reproduzíveis |

**Entregas (seção 7):** slides (≤ 10) ✓ · notebook executável ✓ · aplicação web funcional com instruções para reiniciá-la a partir do código ✓

**Além do pedido:** verificação analítica do número de parâmetros, gráfico treino × validação para diagnosticar sobre/subajuste, matriz de confusão normalizada, execução determinística e registro da divisão de dados para reprodução exata.
