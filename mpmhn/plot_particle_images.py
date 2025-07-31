import numpy as np
import plotly.express as px

def to_uint8(img):
    """
    正規化してuint8に変換
    """
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
    img = xs#to_uint8(xs)
    for t in range(steps):
        for i in range(num_particles):
            images.append(img[t, i].reshape(32, 32))
            t_list.append(t)
            p_list.append(i)
    images = np.stack(images, axis=0)  # (num_images, 32, 32)

    fig = px.imshow(
        images,
        animation_frame=0,
        labels={"animation_frame": "index"},
        binary_string=False,
        color_continuous_scale="gray"
    )
    steps_labels = [f"t={t},p={p}" for t, p in zip(t_list, p_list)]
    for k, step in enumerate(fig.layout.sliders[0]['steps']):
        step['label'] = steps_labels[k]

    fig.update_layout(title="Particle Images (t, particle)")
    fig.write_html("test_output_express.html")
    fig.show()

def plot_img(image):
    fig = px.imshow(
        image,
        color_continuous_scale="gray"
    )
    fig.update_layout(title="memory image")
    fig.write_html("memory.html")
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
