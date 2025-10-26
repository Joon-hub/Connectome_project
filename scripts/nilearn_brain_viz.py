from brain_pipeline.nilearn_visualizer import NiLearnVisualizer
import pandas as pd

# Load your error maps
rest_errors = pd.read_csv("data/results/error_map_piop2_training_fold1.csv")
task_errors = pd.read_csv("data/results/error_map_piop1_fold1.csv")

# Create visualizer
viz = NiLearnVisualizer()

# Generate brain maps for task condition
viz.create_error_brain_map(task_errors, "gender_task_brain")

# Compare rest vs task
viz.create_network_comparison(rest_errors, task_errors, "rest_vs_task")