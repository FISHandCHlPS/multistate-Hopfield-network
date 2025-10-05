import numpy as np

from plot.loader import extract_data, extract_parameters, results_loader
from plot.similarity import plot_cos_sim_per_param

# results = results_loader(root="output/multi_pattern_mhn/run")
results = results_loader(root="output/multi_pattern_mhn/multirun")
data = extract_data(results, loading_data="history")
print(data.shape)
#print(extract_parameters(results))
#plot_cos_sim_per_param(results, memory=np.eye(2))
