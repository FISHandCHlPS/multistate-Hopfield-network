"""時間変化量の時系列推移をプロットする。"""
from pathlib import Path
from typing import Literal

import plotly.express as px

from plot.evaluation import calc_timechange
from plot.loader import extract_data, extract_parameters
from plot.utils import array2df


def plot_timechange_mean(
    multirun_data: list[dict[Literal["history", "weight", "initial", "config", "run_dir"]]],
    path: str = "output",
    filename: str = "timechange_mean_per_param.html",
) -> None:
    """パラメータ毎に時間変化量の平均をプロットする関数

    Args:
        multirun_data (list[dict]): パラメータ毎の結果データ
        path (str): 出力先のディレクトリ
        filename (str): 出力ファイル名

    """
    # 履歴データを取得 (run, step, particles, dim)
    history_data = extract_data(multirun_data, loading_data="history")
    temporal_mean, _ = calc_timechange(history_data)
    df_mean = array2df(temporal_mean, column_names=["index", "mean_timechange"])

    # パラメータ情報を取得してデータフレームに結合
    params = extract_parameters(multirun_data)
    df_with_params = df_mean.join(params, on="index").drop("index")

    # プロット作成
    fig = px.scatter(
        df_with_params,
        x="beta",  # パラメータとしてbetaを使用（他のパラメータも可）
        y="mean_timechange",
        color="alpha",  # 色分けとして別のカラムを使用
        title="時間変化量の平均（パラメータ毎）",
        labels={
            "beta": "β",
            "mean_timechange": "時間変化量の平均",
            "alpha": "α",
        },
    )

    fig.update_layout(
        xaxis_title="β",
        yaxis_title="時間変化量の平均",
        width=900,
        height=600,
    )

    # 出力ディレクトリの確認と保存
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.write_html(str(Path(path) / filename))
    fig.show()


def plot_timechange_variance(
    multirun_data: list[dict[Literal["history", "weight", "initial", "config", "run_dir"]]],
    path: str = "output",
    filename: str = "timechange_variance_per_param.html",
) -> None:
    """パラメータ毎に時間変化量の分散をプロットする関数

    Args:
        multirun_data (list[dict]): パラメータ毎の結果データ
        path (str): 出力先のディレクトリ
        filename (str): 出力ファイル名

    """
    # 履歴データを取得 (run, step, particles, dim)
    history_data = extract_data(multirun_data, loading_data="history")
    _, temporal_variance = calc_timechange(history_data)
    df_variance = array2df(temporal_variance, column_names=["index", "variance_timechange"])

    # パラメータ情報を取得してデータフレームに結合
    params = extract_parameters(multirun_data)
    df_with_params = df_variance.join(params, on="index").drop("index")

    # プロット作成
    fig = px.scatter(
        df_with_params,
        x="beta",  # パラメータとしてbetaを使用（他のパラメータも可）
        y="variance_timechange",
        color="alpha",  # 色分けとして別のカラムを使用
        title="時間変化量の分散（パラメータ毎）",
        labels={
            "beta": "β",
            "variance_timechange": "時間変化量の分散",
            "alpha": "α",
        },
    )

    fig.update_layout(
        xaxis_title="β",
        yaxis_title="時間変化量の分散",
        width=900,
        height=600,
    )

    # 出力ディレクトリの確認と保存
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.write_html(str(Path(path) / filename))
    fig.show()
