Integrantes: João Victor Alves de Abreu (RM 564946) e Rodrigo Kenshin Viana Matayoshi (RM 564026).

# Territory Vision — Classificação de Imagens de Satélite com CNN

> Global Solution 2026.2 · FIAP · Aprendizado Profundo com Redes Neurais Convolucionais

Rede neural convolucional **treinada do zero** para classificar imagens de satélite do EuroSAT em quatro tipos de uso e cobertura do solo — **Forest**, **Industrial**, **Residential** e **River** —, com seis experimentos comparados pela validação e o melhor modelo publicado em uma **aplicação web Gradio**.

| Entregável | Arquivo |
|---|---|
| Notebook executável (dados, arquiteturas, treino, comparação, avaliação e app) | [`notebooks/Territory_Vision.ipynb`](notebooks/Territory_Vision.ipynb) |
| Apresentação (10 slides) | [`docs/Territory_Vision_Apresentacao_Final.pptx`](docs/Territory_Vision_Apresentacao_Final.pptx) |
| Aplicação web | [`src/app.py`](src/app.py) e última célula do notebook |
| Rubrica × evidências | [`docs/REQUISITOS.md`](docs/REQUISITOS.md) · resultados detalhados em [`docs/RESULTADOS.md`](docs/RESULTADOS.md) |

## Resultados

Notebook executado do início ao fim no GitHub Actions (CPU, Python 3.12, TensorFlow 2.20). Duas rodadas: na primeira (CNN 1–3) cada configuração mudou várias coisas; na segunda (CNN 4–6) partimos da CNN 2 e mudamos **uma única coisa por vez**.

| Experimento | O que muda | Validação | Menor perda (val.) | Épocas | Parâmetros |
|---|---|---:|---:|---:|---:|
| CNN 1 (base) | Conv 16 → 32 · Dense 64 · sem dropout | 84,38 % | 0,448 | 12 | 529.700 |
| **CNN 2 (selecionada)** | **Conv 32 → 64 · Dense 128 · Dropout 0,25** | **88,75 %** | **0,335** | **12** | **2.117.188** |
| CNN 3 | Conv 32 → 64 → 128 · Dropout 0,35 · LR 0,0005 | 82,50 % | 0,416 | 12 | 1.142.468 |
| CNN 4 | CNN 2 + BatchNormalization | 83,75 % | 0,356 | 10 | 2.117.476 |
| CNN 5 | CNN 2 + aumento de dados | 86,88 % | 0,397 | 12 | 2.117.188 |
| CNN 6 | CNN 2 + até 30 épocas | 88,13 % | 0,331 | 20 | 2.117.188 |

**Teste (160 imagens, avaliado uma única vez com a CNN 2):** acurácia **85,6 %** · precisão macro 0,856 · recall macro 0,856 · F1 macro 0,856.

| Classe | Precisão | Recall | F1 |
|---|---:|---:|---:|
| Forest | 0,949 | 0,925 | 0,937 |
| Industrial | 0,825 | 0,825 | 0,825 |
| Residential | 0,789 | 0,750 | 0,769 |
| River | 0,860 | 0,925 | 0,892 |

**Leitura dos experimentos**
- **Mais filtros** (CNN 1 → 2) foi a mudança mais eficaz: +4,4 p.p. de acurácia e −25 % na perda de validação.
- **Regularização demais** (CNN 3) causou subajuste: o treino ficou em 84 %, abaixo das redes 1 e 2.
- **BatchNorm** (CNN 4) deixou a validação instável e não trouxe ganho nesta rede rasa com 480 imagens.
- **Aumento de dados** (CNN 5) eliminou o sobreajuste — treino abaixo da validação — e é o caminho mais promissor com mais épocas.
- **Mais épocas** (CNN 6) melhoraram a perda até a 15ª época e depois começou o sobreajuste; a acurácia ficou igual à da CNN 2.
- A maior confusão é **Industrial × Residential** (14 dos 23 erros de teste). O app acertou 19 de 20 imagens de teste no teste automático.

## Dados e método

- **Base:** [EuroSAT RGB](https://github.com/phelber/EuroSAT) (versão oficial no [Zenodo 7711810](https://zenodo.org/records/7711810)), 200 imagens por classe sorteadas com semente 42.
- **Preparação:** RGB, 64 × 64, pixels normalizados para [0, 1]; divisão estratificada 60/20/20 (480/160/160), registrada em `outputs/split.json`.
- **Modelos:** Conv2D (3 × 3, ReLU) + MaxPooling2D, Flatten, Dense (ReLU), Dropout, BatchNormalization ou aumento de dados opcionais e saída softmax de 4 classes; **sem modelos pré-treinados**.
- **Treino:** Adam, *sparse categorical cross-entropy*, lote 32, até 12 épocas (30 na CNN 6), EarlyStopping com restauração dos melhores pesos pela perda de validação, execução determinística.
- **Seleção:** maior acurácia de validação (desempate pela menor perda). O teste fica reservado para a avaliação final.

## Como executar

### Google Colab (recomendado)

1. Abra `notebooks/Territory_Vision.ipynb` no Colab.
2. Execute as células em ordem (*Ambiente de execução → Executar tudo*). A primeira instala as dependências; o treino baixa o EuroSAT (~91 MB) e leva poucos minutos em CPU.
3. A célula da seção 6 inicia o Gradio e mostra um **link público temporário** (`*.gradio.live`). Abra o link, envie uma imagem e clique em **Classificar**.

### Localmente

Requisitos: Python 3.10–3.12.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/train.py                  # baixa o EuroSAT, treina as 6 CNNs e gera outputs/
python src/app.py                    # abre em http://127.0.0.1:7860
```

O aplicativo exibe a imagem enviada, a classe prevista e a pontuação de cada uma das quatro classes. Ele precisa de `outputs/best.keras`, gerado pelo treino (os pesos não são versionados por tamanho).

## Estrutura

```text
notebooks/Territory_Vision.ipynb   notebook completo, executado
src/train.py                       preparação, treino dos 6 experimentos, seleção e avaliação
tests/smoke_app.py                 teste automático do app com imagens reais
.github/workflows/train.yml        executa o notebook inteiro e testa o app a cada push
src/app.py                         aplicação Gradio
outputs/                           métricas (results.json), divisão (split.json), curvas e matriz de confusão
docs/                              apresentação, rubrica e resultados detalhados
requirements.txt                   dependências
```

## Limitações

- Amostra de 800 imagens e uma única divisão; as diferenças de 1 a 3 imagens entre CNN 2, 5 e 6 não são conclusivas.
- A divisão aleatória pode colocar imagens de áreas vizinhas em conjuntos diferentes; os resultados não garantem desempenho em outras regiões ou sensores.
- As pontuações softmax não são probabilidades calibradas.
- A classe River indica a presença visual de um rio — não mede volume nem disponibilidade de água.

## Uso de IA generativa

Usamos IA generativa como ferramenta de apoio (dúvidas, sugestões de código, depuração e revisão de textos). As sugestões foram avaliadas pelos resultados executados, e todos os números deste README foram conferidos em `outputs/results.json`. Detalhes na seção 7 do notebook.
