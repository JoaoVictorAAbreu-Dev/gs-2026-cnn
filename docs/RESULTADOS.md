# Experimentos executados — 05/11/2026

Fonte: EuroSAT RGB oficial, Zenodo registro 7711810. Arquivo baixado com SHA-256 `b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90`. Uma amostra de 800 imagens, balanceada entre as quatro classes, foi dividida de forma estratificada em 480 treino, 160 validação e 160 teste. Semente: 42. Execução local em CPU, TensorFlow 2.20 e Python 3.12.

| Modelo | Acurácia validação | Épocas | Parâmetros |
|---|---:|---:|---:|
| CNN 1 | 0,8500 | 12 | 529.700 |
| CNN 2 | 0,86875 | 12 | 2.117.188 |
| CNN 3 | 0,83125 | 10 | 1.142.468 |

A CNN 2 foi selecionada usando somente a validação. No teste reservado, a acurácia foi **0,85**, precisão macro 0,8502, recall macro 0,8500 e F1 macro 0,8485. Os resultados completos estão em `outputs/results.json`, com curvas e matriz de confusão em `outputs/`.

No teste, foram corretamente classificados 37/40 Forest, 32/40 Industrial, 29/40 Residential e 38/40 River. A confusão mais frequente foi Industrial classificado como Residential (8 exemplos); o inverso ocorreu em 6. Residential teve 5 imagens classificadas como River. O treino da CNN 2 atingiu acurácia maior que a validação, indicando diferença a investigar. O experimento 3 foi inferior em validação nesta execução; como várias escolhas mudaram simultaneamente, isso não prova qual delas causou a diferença. Uma única partição e semente limitam a certeza da comparação.

A aplicação Gradio foi iniciada localmente e uma imagem real de Forest foi enviada à rota de classificação pelo cliente Gradio: retornou as quatro probabilidades, com Forest como classe mais provável. Isso comprova uma chamada local, não o desempenho em novas regiões. A divisão aleatória pode colocar imagens espacialmente próximas em conjuntos diferentes; validar por região e por sensor, além de calibrar probabilidades, são trabalhos futuros. A classificação visual River não estima escassez de água.

Uso de IA generativa: assistente de programação ajudou a escrever e executar código e documentos; a dupla deve revisar, compreender e declarar o uso real no material. A identificação dos alunos e RMs consta nos slides. Nenhuma submissão foi feita.

Validação de reprodutibilidade: notebook executado integralmente em diretório separado, com mesma arquitetura selecionada, acurácias de validação e matriz de teste. As saídas do treino e gráficos foram preservados no notebook; a célula do app precisa ser executada novamente para abrir o servidor.
