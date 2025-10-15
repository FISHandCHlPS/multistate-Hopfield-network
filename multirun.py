from plot.loader import results_loader
from plot.similarity import plot_cos_multirun

multirun_path = "output/multi_pattern_mhn/multirun/2025-10-16/00-45-26"
results = results_loader(root=multirun_path)
plot_cos_multirun(results, memory=results[0]["weight"])
