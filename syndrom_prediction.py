# IMPORTANDO MODULOS
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score,
    roc_curve, auc
)
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier


# DEFINICAO DE FUNCOES
# validacao cruzada
def cross_validation(modelo, X, y, n_splits=10):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acuracias, f1s, aucs, top3s = [], [], [], []

    # y_bin e usado apenas para calcular o AUC multiclasse
    y_bin = label_binarize(y, classes=np.unique(y))

    for fold, (idx_treino, idx_validacao) in enumerate(kfold.split(X, y)):
        X_treino, y_treino = X[idx_treino], y[idx_treino]
        X_validacao, y_validacao = X[idx_validacao], y[idx_validacao]
        y_bin_val = y_bin[idx_validacao]

        modelo.fit(X_treino, y_treino)
        y_pred = modelo.predict(X_validacao)
        y_prob = modelo.predict_proba(X_validacao)

        acc = accuracy_score(y_validacao, y_pred)
        f1 = f1_score(y_validacao, y_pred, average="weighted")
        auc_val = roc_auc_score(y_bin_val, y_prob, multi_class="ovr", average="macro")
        top3 = top_k_accuracy_score(y_validacao, y_prob, k=3)

        acuracias.append(acc)
        f1s.append(f1)
        aucs.append(auc_val)
        top3s.append(top3)

        print(f"Fold {fold+1}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc_val:.4f}, Top3={top3:.4f}")

    resultados = pd.DataFrame({
        "Acuracia Media": [np.mean(acuracias)],
        "F1 Medio": [np.mean(f1s)],
        "AUC Medio": [np.mean(aucs)],
        "Top-3 Accuracy": [np.mean(top3s)]
    })

    print("\nResultados Medios:")
    print(resultados)
    return modelo, resultados


# curva ROC media
def plotar_roc_media(modelo, X, y, n_splits=10, titulo="ROC Media"):
    y_bin = label_binarize(y, classes=np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []

    print(f"\nGerando ROC media: {titulo}")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        y_bin_val = y_bin[val_idx]

        modelo.fit(X_train, y_train)
        y_prob = modelo.predict_proba(X_val)

        fpr, tpr, _ = roc_curve(y_bin_val.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, label=f"{titulo} (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()


# CARREGAMENTO DO DATASET
with open("mini_gm_public_v0.1.p", "rb") as f:
    data = pickle.load(f)

# ANALISANDO DATASET
print(f"Quantidade de sindromes: {len(data)}")

X, y = [], []
for sindrome, subjects in data.items():
    for subject, images in subjects.items():
        for image_id, array in images.items():
            X.append(array)
            y.append(sindrome)

X_np = np.vstack(X)
y_np = np.array(y)

# cria dataframe com embeddings e rotulos
df = pd.DataFrame(X_np, columns=[f"X{i}" for i in range(X_np.shape[1])])
df["sindrome"] = y_np
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Numero de amostras: {len(df_shuffled)}")
print("Imagens por sindrome:")
print(df_shuffled["sindrome"].value_counts())

# informacoes sobre o dataset
print("\nInformacoes sobre os dados:")
print(df_shuffled.iloc[:, :-1].info())
print(df_shuffled.iloc[:, -1].info())

# verifica se existe nulo
if df_shuffled.iloc[:, :-1].isnull().sum().sum() > 0:
    print("Existe valor nulo")
else:
    print("Nao existe valor nulo")

# codifica as classes em numeros inteiros
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(df_shuffled["sindrome"])
X = df_shuffled.iloc[:, :-1].to_numpy()

# ANALISE T-SNE
X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)

plt.figure(figsize=(10,8))
scatter = plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_encoded, cmap="tab20", alpha=0.7)
handles, _ = scatter.legend_elements()
labels = encoder.classes_
plt.legend(handles, labels, title="Sindrome", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# segunda variacao de parametros para TSNE
X_embedded = TSNE(
    n_components=2,
    perplexity=50,
    learning_rate=1000,
    n_iter=4000,
    init="pca",
    random_state=42,
    method="exact"
).fit_transform(X)

plt.figure(figsize=(10,8))
scatter = plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_encoded, cmap="tab20", alpha=0.7)
handles, _ = scatter.legend_elements()
plt.legend(handles, labels, title="Sindrome", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# CLASSIFICATION TASK AND METRICS AND EVALUATION
# separa dataset de treino e teste
X_train = X[:1114, :]
X_test = X[1114:1116, :]
y_train = y_encoded[:1114]
y_test = y_encoded[1114:1116]

# ----------- KNN Euclidean
modelo_knn5 = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
final_model_knn5, _ = cross_validation(modelo_knn5, X_train, y_train)

modelo_knn10 = KNeighborsClassifier(n_neighbors=10, metric="euclidean")
final_model_knn10, _ = cross_validation(modelo_knn10, X_train, y_train)

modelo_knn13 = KNeighborsClassifier(n_neighbors=13, metric="euclidean")
final_model_knn13, _ = cross_validation(modelo_knn13, X_train, y_train)

# analise das curvas ROC para euclidean
plt.figure(figsize=(8,6))
plotar_roc_media(modelo_knn5, X_train, y_train, titulo="KNN 5 - Euclidean")
plotar_roc_media(modelo_knn10, X_train, y_train, titulo="KNN 10 - Euclidean")
plotar_roc_media(modelo_knn13, X_train, y_train, titulo="KNN 13 - Euclidean")
plt.plot([0,1],[0,1],'--',color='gray')
plt.show()

# ------------ KNN Cosine 
modelo_cosine5 = KNeighborsClassifier(n_neighbors=5, metric="cosine")
final_model_cosine5, _ = cross_validation(modelo_cosine5, X_train, y_train)

modelo_cosine10 = KNeighborsClassifier(n_neighbors=10, metric="cosine")
final_model_cosine10, _ = cross_validation(modelo_cosine10, X_train, y_train)

modelo_cosine13 = KNeighborsClassifier(n_neighbors=13, metric="cosine")
final_model_cosine13, _ = cross_validation(modelo_cosine13, X_train, y_train)

# analise das curvas ROC para cosine
plt.figure(figsize=(8,6))
plotar_roc_media(modelo_cosine5, X_train, y_train, titulo="KNN 5 - Cosine")
plotar_roc_media(modelo_cosine10, X_train, y_train, titulo="KNN 10 - Cosine")
plotar_roc_media(modelo_cosine13, X_train, y_train, titulo="KNN 13 - Cosine")
plt.plot([0,1],[0,1],'--',color='gray')
plt.show()

# comparacao entre as duas melhores curvas
plt.figure(figsize=(8,6))
plotar_roc_media(modelo_cosine10, X_train, y_train, titulo="KNN 10 - Cosine")
plotar_roc_media(modelo_knn13, X_train, y_train, titulo="KNN 13 - Euclidean")
plt.plot([0,1],[0,1],'--',color='gray')
plt.title("Comparacao das melhores curvas ROC")
plt.show()

# PREDICTION TASK
# avaliacao final no conjunto de teste
print("\nAvaliacao final no conjunto de teste (2 amostras):")
y_pred_euclidean = final_model_knn13.predict(X_test)
y_pred_cosine = final_model_cosine10.predict(X_test)

print(f"y_test: {y_test}")
print(f"y_pred (Euclidean, k=13): {y_pred_euclidean}")
print(f"y_pred (Cosine, k=10): {y_pred_cosine}")
