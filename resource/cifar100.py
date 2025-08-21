"""
CIFAR-100の画像をダウンロードして表示するサンプルコード
"""
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100
from torchvision import transforms
import numpy as np
from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp
from typing import Tuple

# 画像表示用関数
def show_images(images: ArrayLike, labels: ArrayLike, class_names: list[str]) -> None:
    """
    グレースケール画像の配列を、クラスごとにグリッド表示する。

    Args:
        images (ArrayLike): 画像配列。shape=(N, 1, H, W)
        labels (ArrayLike): クラスラベル配列。shape=(N,)
        class_names (list[str]): クラス名リスト。
    Returns:
        None
    """
    n = len(images)
    grid_size = int(np.ceil(np.sqrt(n)))  # 10x10グリッド想定
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*1, grid_size*1))
    for idx, ax in enumerate(axes.flat):
        if idx < n:  # 画像がある場合
            img = images[idx]
            img = img[0]  # (1, H, W) -> (H, W)
            ax.imshow(img, cmap='gray')
            ax.set_title(class_names[labels[idx]], fontsize=8)
            ax.axis('off')
        else:  # 画像がない場合は表示しない（余ったセル）
            ax.axis('off')
    plt.subplots_adjust(
        left=0.02,    # 左側の余白
        bottom=0.02,  # 下側の余白
        right=0.98,   # 右側の余白
        top=0.95,     # 上側の余白（タイトル分少し余裕）
        wspace=0.05,  # 列間の間隔
        hspace=0.15   # 行間の間隔（タイトル分少し広め）
    )
    plt.show()


def pytorch_to_jax(tensor_batch: 'torch.Tensor') -> Array:
    """
    PyTorchのテンソルをJAXのndarrayに変換する。

    Args:
        tensor_batch (torch.Tensor): PyTorchのテンソル
    Returns:
        Array: JAXのndarray
    """
    numpy_batch = tensor_batch.detach().cpu().numpy()
    jax_batch = jnp.array(numpy_batch)
    return jax_batch


def get_cifar100() -> Tuple[Array, Array, list[str]]:
    """
    CIFAR-100データセットから、全100クラスそれぞれ1枚ずつグレースケール画像を取得する。

    Returns:
        images (Array): 画像配列（shape: [100, 1, 32, 32]、JAX配列）。
        labels (Array): クラスラベル配列（shape: [100]、JAX配列）。
        class_names (list[str]): クラス名リスト（長さ100）。

    処理の流れ:
        1. torchvisionのCIFAR-100データセットをグレースケール変換付きでロード
        2. 各クラスから最初の1枚だけ画像を抽出
        3. クラス順に画像・ラベル・クラス名を返す
    """
    # CIFAR-100データセットをロード
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),    # グレースケール変換
        transforms.ToTensor(),                           # テンソル変換
    ])
    dataset = CIFAR100(root='resource/data', train=True, download=True, transform=transform)
    class_names = dataset.classes

    # 各クラスから1枚ずつ画像を取得
    class_to_img = {}
    for img, label in dataset:
        if label not in class_to_img:
            class_to_img[label] = (pytorch_to_jax(img), label)
        if len(class_to_img) == len(class_names):
            break
    # クラス順に並べる
    sorted_items = sorted(class_to_img.items(), key=lambda x: x[0])
    images = jnp.stack([item[1][0] for item in sorted_items], axis=0)
    labels = jnp.array([item[1][1] for item in sorted_items])

    images = normalize(images)  # 標準化
    return images, labels, class_names


def normalize(images: Array) -> Array:
    """
    画像配列を正規化する。
    """
    images = jnp.asarray(images)
    mean = images.mean(axis=(1, 2, 3), keepdims=True)
    std = images.std(axis=(1, 2, 3), keepdims=True)
    return (images - mean) / (std + 1e-10)


if __name__ == '__main__':
    """
    CIFAR-100の全クラスからグレースケール画像を1枚ずつ取得し、
    クラス名付きでグリッド表示するデモスクリプト。
    1. get_cifar100() で画像・ラベル・クラス名を取得
    2. show_images() で可視化
    """
    images, labels, class_names = get_cifar100()
    show_images(images, labels, class_names)
