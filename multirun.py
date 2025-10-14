from plot.loader import results_loader
from plot.similarity import plot_cos_multirun

results = results_loader(root="output/multi_pattern_mhn/multirun")
plot_cos_multirun(results, memory=results[0]["weight"])
