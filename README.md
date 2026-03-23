# AlexNet — Stanford Dogs Classification

Atividade 3 da disciplina Tópicos para Computação 1 (2026.1) — Escola Superior de Tecnologia (UEA).

Implementação do zero da arquitetura AlexNet para classificação multiclasse do [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), com 120 raças de cachorro e cerca de 20 mil imagens.

---

## Dataset

- **Stanford Dogs Dataset** via Kaggle: [miljan/stanford-dogs-dataset-traintest](https://www.kaggle.com/datasets/miljan/stanford-dogs-dataset-traintest)
- 120 classes (raças de cachorro)
- Imagens organizadas em pastas `train/` e `test/`

---

## Ambiente

O projeto usa [uv](https://github.com/astral-sh/uv) para gerenciamento de dependências.

```bash
uv sync
```

Para registrar o kernel no Jupyter:

```bash
uv run python -m ipykernel install --user --name=alexnet-cnn-training
```

Dependências principais: `torch`, `torchvision`, `scikit-learn`, `seaborn`, `matplotlib`, `pandas`, `tqdm`.

---

## Configuração do Kaggle

Crie um arquivo `.env` na raiz do projeto com suas credenciais:

```
KAGGLE_USERNAME=seu_usuario
KAGGLE_KEY=sua_chave
```

A chave pode ser gerada em [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token.

---

## Arquitetura

AlexNet implementada camada a camada em PyTorch:

| Camada | Tipo | Detalhe |
|--------|------|---------|
| Conv1 | Conv2d + ReLU + MaxPool | 96 filtros 11×11, stride 4 |
| Conv2 | Conv2d + ReLU + MaxPool | 256 filtros 5×5, padding 2 |
| Conv3 | Conv2d + ReLU | 384 filtros 3×3 |
| Conv4 | Conv2d + ReLU | 384 filtros 3×3 |
| Conv5 | Conv2d + ReLU + MaxPool | 256 filtros 3×3 |
| Adaptação | AdaptiveAvgPool2d | saída 6×6 |
| FC6 | Linear + ReLU + Dropout(0.5) | 4096 neurônios |
| FC7 | Linear + ReLU + Dropout(0.5) | 4096 neurônios |
| FC8 | Linear | 120 saídas |

Total de parâmetros: **58.772.984** (todos treináveis).

---

## Treinamento

| Parâmetro | Valor |
|-----------|-------|
| Épocas | 120 |
| Otimizador | SGD com Momentum |
| Learning rate | 1e-3 |
| Momentum | 0.9 |
| Batch size | 32 |
| Função de perda | CrossEntropyLoss |
| Hardware | GPU (CUDA) |

Métricas salvas em `training_metrics.csv` a cada época.

---

## Resultados

| Conjunto | Acurácia |
|----------|----------|
| Treino | 99,24% |
| Teste | 25,84% |

A diferença expressiva entre treino e teste indica overfitting — esperado dado o volume de parâmetros frente ao tamanho do dataset. O F1 macro no teste ficou em 0,24.

---

## Estrutura

```
.
├── Topicos1-2026.1-Tarefa3.ipynb   # notebook principal
├── training_metrics.csv            # loss e acurácia por época
├── alexnet_model.pth               # pesos salvos (state_dict)
├── modelo.pt                       # modelo completo serializado
├── data/                           # dataset (não versionado)
├── pyproject.toml
└── uv.lock
```

---

## Referências

- Krizhevsky, A., Sutskever, I., Hinton, G. (2012). *ImageNet Classification with Deep Convolutional Neural Networks*. NeurIPS.
- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [PyTorch Documentation](https://docs.pytorch.org)


**README feito com IA**