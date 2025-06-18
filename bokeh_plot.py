from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Category10
import numpy as np

# history の shape: (steps+1, num_particles, 2)
def plot_trajectories(history: np.ndarray):
    """
    2次元空間上で各粒子の軌跡を表示する。
    Args:
        history (np.ndarray): shape=(steps+1, num_particles, 2)
    """
    steps, num_particles = history.shape[0], history.shape[1]
    p = figure(title="Particle Trajectories (Bokeh)", x_axis_label="x0", y_axis_label="x1", width=600, height=500)
    colors = Category10[10]
    for i in range(num_particles):
        x = history[:, i, 0]
        y = history[:, i, 1]
        p.line(x, y, legend_label=f"particle {i}", line_width=2, color=colors[i % len(colors)])
        p.circle(x, y, size=5, color=colors[i % len(colors)], alpha=0.7)
    p.legend.location = "top_left"
    #output_file("particle_trajectories.html")
    show(p)

if __name__ == "__main__":
    from toy import history
    # jax.numpy から numpy へ変換
    plot_trajectories(np.array(history))
