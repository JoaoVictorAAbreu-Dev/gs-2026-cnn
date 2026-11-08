Integrantes: João Victor Alves de Abreu (RM 564946) e Rodrigo Kenshin Viana Matayoshi (RM 564026).

# Territory Vision — Classificação de Imagens de Satélite com CNN

> Global Solution 2026.2 · FIAP · Aprendizado Profundo com Redes Neurais Convolucionais

Rede neural convolucional **treinada do zero** para classificar imagens de satélite do EuroSAT em quatro tipos de uso e cobertura do solo — **Forest**, **Industrial**, **Residential** e **River** —, com três arquiteturas comparadas pela validação e o melhor modelo publicado em uma **aplicação web Gradio**.

| Entregável | Arquivo |
|---|---|
| Notebook executável (dados, arquiteturas, treino, comparação, avaliação e app) | [`notebooks/Territory_Vision.ipynb`](notebooks/Territory_Vision.ipynb) |
| Apresentação (10 slides) | [`docs/Territory_Vision_Apresentacao_Final.pptx`](docs/Territory_Vision_Apresentacao_Final.pptx) |
| Aplicação web | [`src/app.py`](src/app.py) e última célula do notebook |
| Rubrica × evidências | [`docs/REQUISITOS.md`](docs/REQUISITOS.md) · resultados detalhados em [`docs/RESULTADOS.md`](docs/RESULTADOS.md) |

## Resultados

| Experimento | Arquitetura | Validação | Menor perda (val.) | Parâmetros |
|---|---|---:|---:|---:|
| CNN 1 (base) | Conv 16 → 32 · Dense 64 · sem dropout · LR 0,001 | 85,00 % | 0,437 | 529.700 |
| **CNN 2 (selecionada)** | **Conv 32 → 64 · Dense 128 · Dropout 0,25 · LR 0,001** | **86,88 %** | **0,331** | **2.117.188** |
| CNN 3 | Conv 32 → 64 → 128 · Dense 128 · Dropout 0,35 · LR 0,0005 | 83,13 % | 0,451 | 1.142.468 |

**Teste (160 imagens, avaliado uma única vez com a CNN 2):** acurácia **85,0 %** · precisão macro 0,850 · recall macro 0,850 · F1 macro 0,849.

| Classe | Precisão | Recall | F1 |
|---|---:|---:|---:|
| Forest | 0,949 | 0,925 | 0,937 |
| Industrial | 0,842 | 0,800 | 0,821 |
| Residential | 0,784 | 0,725 | 0,753 |
| River | 0,826 | 0,950 | 0,884 |

**Leitura dos experimentos**
- Mais filtros e dropout (CNN 1 → CNN 2) trouxeram o maior ganho: +1,9 p.p. de acurácia e −24 % na perda de validação.
- Mais profundidade com regularização forte e passo menor (CNN 3) causou **subajuste** em 12 épocas: na melhor época, a acurácia de treino (76,7 %) ficou abaixo da de validação (83,1 %).
- A maior confusão é **Industrial × Residential** (14 dos 24 erros de teste), classes com telhados e malha viária parecidos em 64 × 64 pixels.
- A perda de validação da CNN 2 ainda caía na última época — treinar por mais épocas é o próximo passo mais promissor.

## Dados e método

- **Base:** [EuroSAT RGB](https://github.com/phelber/EuroSAT) (versão oficial no [Zenodo 7711810](https://zenodo.org/records/7711810)), 200 imagens por classe sorteadas com semente 42.
- **Preparação:** RGB, 64 × 64, pixels normalizados para [0, 1]; divisão estratificada 60/20/20 (480/160/160), registrada em `outputs/split.json`.
- **Modelos:** Conv2D (3 × 3, ReLU) + MaxPooling2D, Flatten, Dense (ReLU), Dropout opcional e saída softmax de 4 classes; **sem modelos pré-treinados**.
- **Treino:** Adam, *sparse categorical cross-entropy*, lote 32, até 12 épocas, EarlyStopping (paciência 3, melhores pesos pela perda de validação), execução determinística.
- **Seleção:** maior acurácia de validação (desempate pela menor perda). O teste fica reservado para a avaliação final.

## Como executar

### Google Colab (recomendado)

1. Abra `notebooks/Territory_Vision.ipynb` no Colab.
2. Execute as células em ordem (*Ambiente de execução → Executar tudo*). A primeira instala as dependências; o treino baixa o EuroSAT (~91 MB) e leva alguns minutos em CPU.
3. A última célula inicia o Gradio e mostra um **link público temporário** (`*.gradio.live`). Abra o link, envie uma imagem e clique em **Classificar**.

### Localmente

Requisitos: Python 3.10–3.12.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/train.py                  # baixa o EuroSAT, treina as 3 CNNs e gera outputs/
python src/app.py                    # abre em http://127.0.0.1:7860
```

O aplicativo exibe a imagem enviada, a classe prevista e a pontuação de cada uma das quatro classes. Ele precisa de `outputs/best.keras`, gerado pelo treino (os pesos não são versionados por tamanho).

## Estrutura

```text
notebooks/Territory_Vision.ipynb   notebook completo, executado
src/train.py                       preparação, treino dos 3 experimentos, seleção e avaliação
src/app.py                         aplicação Gradio
outputs/                           métricas (results.json), divisão (split.json), curvas e matriz de confusão
docs/                              apresentação, rubrica e resultados detalhados
requirements.txt                   dependências
```

## Limitações

- Amostra de 800 imagens e uma única semente; outras divisões podem mudar os números.
- A divisão aleatória pode colocar imagens de áreas vizinhas em conjuntos diferentes; os resultados não garantem desempenho em outras regiões ou sensores.
- As pontuações softmax não são probabilidades calibradas.
- A classe River indica a presença visual de um rio — não mede volume nem disponibilidade de água.

## Uso de IA generativa

Usamos IA generativa como ferramenta de apoio (dúvidas, sugestões de código, depuração e revisão de textos). As sugestões foram avaliadas pelos resultados executados, e todos os números deste README foram conferidos em `outputs/results.json`. Detalhes na seção 7 do notebook.
