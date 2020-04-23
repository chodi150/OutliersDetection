import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA

def show_pca(data_frame, outliers_indices, entity_names):
    pca = PCA(n_components=2)
    pca_df = pca.fit_transform(data_frame)
    pca_df = pd.DataFrame(data = pca_df, columns = ['c1', 'c2'])
    pca_df["type"] = ["red" if x in outliers_indices else "blue" for x in range(len(data_frame))]
    pca_df["entity_name"] = entity_names
    #plt.scatter(principalDf.iloc[:, 0], principalDf.iloc[:, 1], c=y_kmeans, cmap='viridis')
    fig = go.Figure(data=go.Scatter(x = np.array(pca_df["c1"]), y=pca_df["c2"], text=pca_df["entity_name"], mode = "markers" ,marker=dict(
            color=np.array(np.array(pca_df["type"])),
            colorscale='Viridis',
        )))

    fig.update_layout( title="PCA")

    fig.show()

def show_tsne(data_frame, outliers_indices, entity_names):
    df_tsne = TSNE(n_components=2, perplexity = 12, learning_rate = 50, n_iter = 5000).fit_transform(data_frame)
    df_tsne = pd.DataFrame(data = df_tsne, columns = ['c1', 'c2'])
    df_tsne["type"] = ["red" if x in outliers_indices else "blue" for x in range(len(data_frame))]
    df_tsne["entity_name"] = entity_names
    fig = go.Figure(data=go.Scatter(x = np.array(df_tsne["c1"]), y=df_tsne["c2"], text=df_tsne["entity_name"], mode = "markers" ,marker=dict(
            color=np.array(np.array(df_tsne["type"])),
            colorscale='Viridis',
        )))

    fig.update_layout(title="t-SNE")
    fig.show()