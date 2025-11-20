from plotly import io as pio

from plot.entropy import plot_entropy_multirun
from plot.loader import results_loader

# VSCode用のレンダラーを設定
pio.renderers.default = "browser"

multirun_path = "output/multi_pattern_mhn/multirun/2025-11-13/17-01-22"
results = results_loader(root=multirun_path)
plot_entropy_multirun(results, memory=results[0]["weight"])
