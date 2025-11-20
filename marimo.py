import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from jax.scipy.stats import gamma

    # 平均 mu, 標準偏差 std
    mu = 0.3
    std = 0.3

    def normal(x, mu, sigma):
        pdf = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-( (x - mu)**2 ) / (2 * sigma**2) )
        return np.where((1e-3 < x) & (x < 5), pdf, 0)

    # 正規分布の生成
    x = np.linspace(0, 1, 1000)
    # y0 = normal(x, mu=0.3, sigma=0.3)
    # y1 = normal(x, mu=0.7, sigma=0.3)
    y0 = gamma.pdf(x, a=10, scale=0.02)     # 平均0.2
    y1 = gamma.pdf(x, a=20, scale=0.03)     # 平均0.3

    # プロット
    plt.style.use("seaborn-v0_8-white")
    plt.plot(x, y0, color='blue', linewidth=2)
    plt.plot(x, y1, color='red', linewidth=2)
    # plt.xlim(-1, 1)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("density", fontsize=14)
    plt.title("Distribution of Speed", fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return np, plt


@app.cell
def _(plt):
    plt.style.available
    return


@app.cell
def _(np):
    from resource.cifar100 import get_cifar100
    import matplotlib.pyplot as plot

    images, _, _ = get_cifar100()  # images: (100, 1, 32, 32)
    images_flat = images.reshape(images.shape[0], -1)  # (100, 1024)
    # 各画像を正規化
    images_flat = images_flat / np.linalg.norm(images_flat, axis=1, keepdims=True)

    plot.figure(figsize=(10, 10))
    for i in range(3):
        plot.subplot(3, 1, i+1)
        plot.imshow(images_flat[i].reshape(32, 32), cmap='gray')
        plot.axis('off')
    plot.show()
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.pyplot as pyplot
    from matplotlib.colors import Normalize

    cmap = pyplot.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(2, 6))
    fig.colorbar(ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.08734924, vmax=0.7669708)),
                cax=ax,
                orientation="vertical")  # 縦向き
    plt.show()
    return


if __name__ == "__main__":
    app.run()
