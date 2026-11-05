# Territory Vision

Classificador experimental de imagens de satélite para a Global Solution FIAP 2026.2 de Aprendizado Profundo com CNN. O projeto compara três redes convolucionais treinadas do zero e apresenta a classificação em uma aplicação web local.

**Integrantes:** João Victor Alves de Abreu (RM 564946) e Rodrigo Kenshin Viana Matayoshi (RM 564026).

As quatro classes são **Forest** (floresta), **Industrial** (área industrial), **Residential** (área residencial) e **River** (rio). O modelo identifica padrões visuais em imagens semelhantes às do EuroSAT; não mede volume ou qualidade da água e não prevê escassez hídrica.

## Resultados observados

O experimento local usou 800 imagens RGB de 64 × 64 pixels, distribuídas igualmente entre as quatro classes. A amostra foi dividida de forma estratificada em 480 imagens de treino, 160 de validação e 160 de teste, com semente 42.

| Experimento | Acurácia na validação | Parâmetros |
| --- | ---: | ---: |
| CNN 1 | 85,00% | 529.700 |
| **CNN 2** | **86,88%** | **2.117.188** |
| CNN 3 | 83,13% | 1.142.468 |

A CNN 2 foi escolhida pela maior acurácia de validação; em caso de empate, o código usa a menor perda de validação. No conjunto de teste reservado, ela obteve **85,00% de acurácia**, precisão macro de **0,8502**, recall macro de **0,8500** e F1 macro de **0,8485**. A acurácia mede a proporção total de acertos; as médias macro dão o mesmo peso a cada classe.

No teste, as confusões mais frequentes envolveram as classes industrial e residencial. A análise por classe, a matriz de confusão e as curvas estão em [Resultados](docs/RESULTADOS.md) e nos arquivos gerados em `outputs/`. As três configurações modificam mais de uma escolha por vez; a comparação não isola a causa de uma diferença de desempenho.

## Dados e método

O treinamento baixa a versão RGB do [EuroSAT no Zenodo](https://zenodo.org/records/7711810), seleciona 200 imagens por classe com semente fixa, redimensiona as imagens para 64 × 64 e normaliza os pixels para o intervalo de 0 a 1. O arquivo baixado na execução registrada teve SHA-256 `b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90`.

Cada modelo combina camadas `Conv2D`, ReLU, pooling e camadas densas, com saída softmax de quatro classes. As configurações variam quantidade de filtros e camadas, tamanho da camada densa, dropout e taxa de aprendizado. O treino usa Adam, entropia cruzada esparsa e parada antecipada. A avaliação final usa o conjunto de teste somente após a seleção pela validação.

O arquivo `outputs/split.json` registra os caminhos das imagens usados em cada conjunto. A mesma semente e a execução repetida localmente reproduziram a matriz de teste documentada; versões de biblioteca e ambiente podem afetar a reprodução numérica. Confira os termos da fonte antes de redistribuir o dataset.

## Executar localmente

**Pré-requisitos:** Python 3.12, [uv](https://docs.astral.sh/uv/), conexão para baixar o dataset e instalar dependências, e espaço para o arquivo RGB (aproximadamente 91 MiB), imagens extraídas, bibliotecas e modelos. Os comandos devem ser executados na raiz do projeto:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python src/train.py
.venv/bin/python src/app.py
```

O primeiro treino baixa e extrai o EuroSAT quando necessário e gera os modelos, gráficos e métricas. Ele pode levar alguns minutos em CPU. Depois, abra no navegador o endereço local mostrado pelo Gradio, normalmente `http://127.0.0.1:7860`, envie uma imagem e clique em **Classificar**. O aplicativo mostra a imagem e a pontuação das quatro classes. O app precisa de `outputs/best.keras`; se o arquivo não existir, execute o treinamento primeiro.

Se o Python 3.12 estiver instalado no PATH e você preferir `venv` e `pip`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/train.py
.venv/bin/python src/app.py
```

O notebook [Territory Vision](notebooks/Territory_Vision.ipynb) reúne preparação, treino, comparação, avaliação e aplicação. Execute as células em ordem a partir de um ambiente Python compatível, como o Google Colab. A célula da aplicação deve ser executada novamente para iniciar seu servidor. O notebook foi executado integralmente em uma cópia isolada e reproduziu a seleção e a matriz de teste registradas.

## Estrutura

```text
src/train.py                    Preparação, treino, seleção e avaliação
src/app.py                      Aplicação Gradio
notebooks/Territory_Vision.ipynb Notebook com código, análise e gráficos
docs/REQUISITOS.md              Rubrica e situação dos itens da prova
docs/RESULTADOS.md              Métricas, análise e evidências locais
docs/                            Apresentação da equipe
requirements.txt                Dependências Python
data/                            Dataset baixado e extraído, gerado localmente
outputs/                         Divisão, métricas, gráficos e modelos locais
```

O repositório versiona métricas, divisão e gráficos em `outputs/`. O dataset, os pesos `.keras` e `.venv/` ficam fora do Git: em um clone limpo, instale as dependências e execute o treino para obter o modelo necessário ao aplicativo. O teste do aplicativo realizado neste ambiente enviou uma imagem real e recebeu as quatro probabilidades pela interface Gradio.

## Limitações e uso responsável

- A divisão é aleatória e pode colocar imagens espacialmente próximas em conjuntos distintos. Os números não demonstram generalização para novas regiões ou sensores.
- O estudo usa uma amostra de 800 imagens e uma única semente. Outras amostras e divisões podem produzir resultados diferentes.
- As saídas softmax são pontuações do modelo e não foram calibradas como probabilidades confiáveis. A aplicação não substitui avaliação especializada.
- Imagens de fora do EuroSAT, com outra resolução, estação ou sensor, podem receber classificações incorretas. A presença visual da classe River não informa disponibilidade hídrica.

## Entrega acadêmica

A rubrica e as evidências de cada critério estão em [REQUISITOS.md](docs/REQUISITOS.md). João e Rodrigo afirmam ter conhecimento para desenvolver grande parte da solução de forma independente. Neste projeto, utilizaram apoio de IA generativa na organização dos experimentos, na elaboração e execução de código e na documentação. A dupla deve revisar os resultados, compreender as decisões e responder pelo trabalho entregue.

Antes de submeter, a dupla precisa revisar notebook, app e slides, e confirmar prazo e formato no Portal do Aluno. A entrega ainda não foi submetida.
