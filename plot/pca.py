"""
PCAによる画像の可視化
"""
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA

def plot_pca_images(image_array: np.ndarray, path: str='output', filename: str='pca.html') -> None:
    """
    画像配列をPCAで次元削減し、散布図としてプロットする。

    Args:
        image_array (np.ndarray): 画像配列 (画像数, 画像1枚の次元数)
        path (str, optional): 出力先のパス。Defaults to 'output'。
        filename (str, optional): 出力ファイル名。Defaults to 'pca.html'。

    Returns:
        None
    """
    pca = PCA()
    reduced = pca.fit_transform(image_array)
    explained_var = pca.explained_variance_ratio_
    print(f"寄与率: {explained_var}")
    print(f"第二主成分までの累積寄与率: {np.sum(explained_var[:2])}")
    df = pd.DataFrame(reduced, columns=[f'PC{i+1}' for i in range(reduced.shape[1])])
    fig = px.scatter(df, x='PC1', y='PC2', title='PCA of Images', width=800, height=600)
    fig.write_html(f"{path}/{filename}")
    fig.show()


def plot_pca_images_trajectory(history: np.ndarray, path: str='output', filename: str='pca_trajectory.html') -> None:
    """
    PCAし、粒子ごとの軌跡を可視化する。

    Args:
        history (np.ndarray): 画像配列 (steps, num_particles, 画像次元)
        path (str, optional): 出力先のパス。Defaults to 'output'。
        filename (str, optional): 出力ファイル名。Defaults to 'pca_trajectory.html'。

    Returns:
        None
    """
    steps, num_particles, dim = history.shape
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(history.reshape(-1, dim))
    explained_var = pca.explained_variance_ratio_
    print(f"第二主成分までの累積寄与率: {np.sum(explained_var[:2])}")
    
    records = []
    for t in range(steps):
        for p in range(num_particles):
            records.append({
                't': t,
                'particle': p,
                'PC1': reduced[t*num_particles + p, 0],
                'PC2': reduced[t*num_particles + p, 1],
            })
    df = pd.DataFrame(records)

    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='particle',
        animation_group='particle',
        animation_frame='t',
        range_x=[-1, 1],
        range_y=[-1, 1],
        title='PCA Trajectory of Particles',
        width=800,
        height=600
    )
    fig.write_html(f"{path}/{filename}")
    fig.show()