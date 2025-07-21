import numpy as np
import plotly.express as px

def to_uint8(img):
    img = np.asarray(img)
    if img.min() < 0 or img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    return img

def plot_particle_image_slider(history):
    """
    plotly.expressで、時刻・粒子ごとにhistoryの各ベクトルを32x32画像として切り替え表示
    history: (steps+1, num_particles, 1024)
    """
    steps, num_particles, dim = history.shape
    xs = np.asarray(history)

    images = []
    t_list = []
    p_list = []
    for t in range(steps):
        for i in range(num_particles):
            img = to_uint8(xs[t, i].reshape(32, 32))
            images.append(img)
            t_list.append(t)
            p_list.append(i)
    images = np.stack(images, axis=0)  # (num_images, 32, 32)

    # 画像の類似度が近い順に並べる（ユークリッド距離の貪欲法）
    def sort_images_by_similarity(images):
        N = images.shape[0]
        flat = images.reshape(N, -1).astype(np.float32)
        used = np.zeros(N, dtype=bool)
        order = [0]
        used[0] = True
        for _ in range(N - 1):
            last = order[-1]
            dists = np.linalg.norm(flat[None, last] - flat, axis=1)
            dists[used] = np.inf
            next_idx = np.argmin(dists)
            order.append(next_idx)
            used[next_idx] = True
        return order

    order = sort_images_by_similarity(images)
    images_sorted = images[order]
    t_list_sorted = [t_list[i] for i in order]
    p_list_sorted = [p_list[i] for i in order]

    fig = px.imshow(
        images_sorted,
        animation_frame=0,  # 0:画像枚数方向
        labels={"animation_frame": "index"},
        binary_string=False,
        color_continuous_scale="gray"
    )
    steps_labels = [f"t={t},p={p}" for t, p in zip(t_list_sorted, p_list_sorted)]
    for k, step in enumerate(fig.layout.sliders[0]['steps']):
        step['label'] = steps_labels[k]

    fig.update_layout(title="Particle Images (t, particle) [sorted by similarity]")
    fig.write_html("test_output_express.html")
    fig.show()




def test_plot_particle_image_slider():
    """
    plot_particle_image_sliderのテスト用関数。
    ダミーデータ（steps=3, num_particles=2, 1024次元）で呼び出し動作を確認。
    """
    steps = 3
    num_particles = 2
    dim = 1024
    np.random.seed(0)
    history = np.random.rand(steps+1, num_particles, dim)
    print('test history.shape:', history.shape)
    plot_particle_image_slider(history)


# テスト実行用
if __name__ == "__main__":
    test_plot_particle_image_slider()
