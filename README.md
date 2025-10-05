# KNNClassification_Syndrome
## KNN Classification - Syndrome Embedding Dataset

### Descricao
Este projeto implementa um pipeline completo de classificacao de sindromes a partir de embeddings de imagens. Para realizar a classificação o mesmo
utiliza o algoritmo K-Nearest Neighbors (KNN) com metricas de distancia Euclidiana e Cosseno.

### Estrutura
- knn_sindrome.py: script principal com todo o pipeline (analise, t-SNE, treinamento e avaliacao)
- requirements.txt: dependencias necessarias
- Report.pdf: relatorio tecnico com metodologia, resultados e analise
- Interpretation.pdf: respostas das questoes interpretativas

### Execucao
1. Instale as dependencias:
   pip install -r requirements.txt

2. Execute o script:
   python knn_sindrome.py

### Observacoes
- O dataset "mini_gm_public_v0.1.p" deve estar no mesmo diretorio do script.
- O script gera automaticamente as curvas ROC e imprime as metricas de desempenho.
