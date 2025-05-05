# ml_toolbox/__init__.py
from .pca_tools import PCATool
from .tsne_tools import TSNETool
from .decision_tree_tools import DecisionTreeTool
from .clustering import ClusteringTool
from .umap_tools import UMAPTool
from .analysis import FeatureAnalysisTool

__all__ = [
    "PCATool",
    "TSNETool",
    "DecisionTreeTool",
    "ClusteringTool",
    "UMAPTool",
    "FeatureAnalysisTool"
]


# ml_toolbox/pca_tools.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class PCATool:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.X_pca = None

    def fit_transform(self, X):
        self.X_pca = self.pca.fit_transform(X)
        return self.X_pca

    def plot_biplot(self, feature_names, target=None):
        coeff = self.pca.components_.T
        xs, ys = self.X_pca[:, 0], self.X_pca[:, 1]
        plt.figure(figsize=(10, 8))
        plt.scatter(xs, ys, c=target, cmap='viridis' if target is not None else None)

        for i, name in enumerate(feature_names):
            plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
            plt.text(coeff[i, 0]*1.2, coeff[i, 1]*1.2, name, color='g')

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Biplot')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ml_toolbox/tsne_tools.py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class TSNETool:
    def __init__(self, n_components=2, perplexity=30, random_state=42):
        self.model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        self.embedding = None

    def fit_transform(self, X):
        self.embedding = self.model.fit_transform(X)
        return self.embedding

    def plot_embedding(self, labels=None, feature=None):
        x, y = self.embedding[:, 0], self.embedding[:, 1]
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x, y, c=labels if feature is None else feature, cmap='viridis')
        if labels is not None:
            plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title("t-SNE Embedding")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ml_toolbox/decision_tree_tools.py
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

class DecisionTreeTool:
    def __init__(self, **kwargs):
        self.model = DecisionTreeClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_

    def plot(self, feature_names):
        plt.figure(figsize=(16, 10))
        plot_tree(self.model, feature_names=feature_names, class_names=self.classes_, filled=True)
        plt.tight_layout()
        plt.show()


# ml_toolbox/clustering.py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class ClusteringTool:
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters)
        self.labels_ = None

    def fit_predict(self, X):
        self.labels_ = self.model.fit_predict(X)
        return self.labels_

    def silhouette(self, X):
        return silhouette_score(X, self.labels_)


# ml_toolbox/umap_tools.py
import matplotlib.pyplot as plt
import umap

class UMAPTool:
    def __init__(self, n_neighbors=15, min_dist=0.1, n_components=2):
        self.model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
        self.embedding = None

    def fit_transform(self, X):
        self.embedding = self.model.fit_transform(X)
        return self.embedding

    def plot_embedding(self, labels=None):
        x, y = self.embedding[:, 0], self.embedding[:, 1]
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x, y, c=labels, cmap='Spectral')
        if labels is not None:
            plt.legend(*scatter.legend_elements(), title="Labels")
        plt.title("UMAP Embedding")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ml_toolbox/analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class FeatureAnalysisTool:
    @staticmethod
    def correlation_matrix(X, feature_names):
        df = pd.DataFrame(X, columns=feature_names)
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def lda_projection(X, y):
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_lda = lda.fit_transform(X, y)
        return X_lda
