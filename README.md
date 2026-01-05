# Tradução Automática EN→PT com Seq2Seq e Attention  
## Entendendo trade-offs em NLP com dados e hardware limitados

Este repositório apresenta um projeto de **tradução automática inglês–português** utilizando **Redes Neurais Recorrentes (LSTM)** com arquitetura **Seq2Seq e mecanismo de Attention**, implementado em **PyTorch**.

O foco do projeto não é alcançar estado da arte, mas **explorar decisões técnicas, limitações práticas e trade-offs reais** envolvendo:

- arquitetura do modelo;
- tamanho do vocabulário;
- volume de dados;
- custo computacional;
- infraestrutura doméstica.

---

## Motivação

Com o domínio atual de **Transformers e modelos de larga escala**, pode parecer irrelevante implementar modelos clássicos de tradução neural. No entanto, essa complexidade abstrai decisões fundamentais que impactam diretamente o comportamento do sistema.

Este projeto parte da seguinte pergunta:

> O que realmente limita a qualidade e a viabilidade de um modelo de tradução quando dados e hardware são escassos?

Ao implementar um sistema completo do zero, foi possível observar na prática como pequenas escolhas — como vocabulário, profundidade da rede ou batch size — afetam desempenho, tempo de treino e estabilidade.

---

## Dataset

- **Europarl Parallel Corpus**
- Pares de sentenças inglês–português
- Domínio: discursos do Parlamento Europeu
- Dataset alinhado e relativamente limpo

Configuração final utilizada:
- ~60k sentenças (antes do filtro)
- ~45k pares após remoção de sentenças com mais de 35 tokens
- Vocabulário limitado a 35k palavras

> ⚠️ **Observação**: Dataset de domínio específico. O modelo não generaliza bem para linguagem cotidiana.

---

## Arquitetura Utilizada

### Seq2Seq com Bahdanau Attention

A arquitetura segue o padrão clássico **Encoder–Decoder**, anterior à popularização dos Transformers.

- **Encoder**: LSTM que processa a sentença em inglês palavra por palavra
- **Attention**: permite que o decoder foque dinamicamente em partes relevantes da sentença fonte
- **Decoder**: LSTM que gera a tradução em português token a token
- **Teacher Forcing** probabilístico durante o treino
- **Beam Search (k=5)** na inferência

A escolha por LSTMs foi intencional, visando:
- maior controle conceitual;
- menor custo computacional;
- melhor compreensão dos mecanismos fundamentais da tradução neural.

---

## Experimentos e Decisões Técnicas

Durante o desenvolvimento, foram avaliados diversos trade-offs:

### Vocabulário
- 20k palavras: treino rápido, muitos `<unk>`
- 35k palavras: melhor equilíbrio entre cobertura e custo
- 50k palavras: treino ~40% mais lento e pior BLEU (overfitting em palavras raras)

### Profundidade do modelo
- 1 camada LSTM: aprendizado estável
- 2 camadas LSTM: pior desempenho (dados insuficientes para maior capacidade)

### Infraestrutura
- GPU: **NVIDIA GTX 1060 – 6GB**
- CPU tornou-se gargalo frequente
- Batch size máximo viável: 64
- Uso de GPU raramente acima de 30%

Tentativas de aumentar o batch size além disso resultaram em **treino mais lento**, não mais rápido, devido ao overhead de memória.

---

## Resultados

Avaliação no conjunto de teste:

| Métrica | Valor |
|------|------|
| BLEU | **10.36** |
| METEOR | **27.01** |

Apesar de um BLEU relativamente baixo quando comparado a sistemas comerciais, o modelo apresentou **bom desempenho dentro do domínio parlamentar**.

### Exemplos corretos (domínio Europarl)
- *i voted for this report* → **votei a favor deste relatório**
- *human rights are important* → **os direitos humanos são importantes**
- *i thank the rapporteur* → **agradeço ao relator**

### Fora do domínio
- *good morning* → `<unk>`
- *hello world* → *no mundo mundial*

Esses exemplos reforçam o caráter **especialista** do modelo: bom desempenho no domínio treinado e baixa generalização fora dele.

---

## Treinamento e Infraestrutura

- Ambiente: **VS Code**
- Hardware:
  - CPU: Intel Core i5 (8ª geração)
  - RAM: 8 GB
  - GPU: NVIDIA GTX 1060 – 6GB

### Tempo de execução
- ~6 minutos por época
- Early stopping com paciência = 3
- Treino completo: ~1h30

---

## Principais Bibliotecas

- Python 3.x
- PyTorch
- TorchText
- NumPy
- Pandas
- Scikit-learn
- NLTK (BLEU / METEOR)

---

## Estrutura do Projeto

- `notebook.ipynb`  
  Notebook principal contendo:
  - pré-processamento do Europarl
  - construção do vocabulário
  - definição do modelo Seq2Seq
  - treinamento com early stopping
  - avaliação com BLEU e METEOR
  - inferência com Greedy Decoding e Beam Search

---

## Conclusões

O projeto mostra que, mesmo longe do estado da arte, modelos clássicos ainda são extremamente valiosos para:

- entender fundamentos de NLP;
- explorar trade-offs reais entre dados, arquitetura e hardware;
- desenvolver intuição prática sobre limitações de modelos neurais.

Mais do que os números finais, o principal ganho foi compreender como decisões aparentemente simples impactam diretamente a qualidade e a viabilidade de um sistema de tradução neural.
