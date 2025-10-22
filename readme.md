# Brain Connectivity Classification Pipeline

A modular Python pipeline for classifying brain regions based on their functional connectivity patterns and detecting regions with altered connectivity during cognitive tasks.

## Project Overview

This pipeline implements a machine learning approach to:
1. **Train** a classifier on resting-state connectivity data (PIOP-2)
2. **Apply** the trained model to task-based data (PIOP-1 gender task)
3. **Identify** brain regions with altered connectivity patterns
4. **Visualize** results and generate comprehensive reports

## Installation

```bash
# Clone or create the project directory
cd brain_connectivity_pipeline

# Install in development mode
pip install -e .

# Or install requirements directly
pip install -r requirements.txt
```

## Project Structure

```
brain_connectivity_pipeline/
├── data/
│   ├── raw/                    # Place your CSV files here
│   ├── processed/              # Intermediate files
│   └── results/                # Output files
├── brain_pipeline/             # Main package
│   ├── config.py              # Configuration management
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessing.py       # Data preprocessing
│   ├── model.py               # Model training and prediction
│   ├── evaluation.py          # Evaluation metrics
│   └── visualization.py       # Plotting functions
├── scripts/                   # Executable scripts
│   ├── 01_train_model.py
│   ├── 02_apply_to_task.py
│   ├── 03_visualize_results.py
│   └── run_full_pipeline.py
└── config.yaml               # Configuration file
```

## Quick Start

### 1. Prepare Your Data

Place your data files in `data/raw/`:
- `PIOP2_restingstate.csv` (resting-state data)
- `PIOP1_gendertask.csv` (task data)

### 2. Run the Complete Pipeline

```bash
python scripts/run_full_pipeline.py
```

This will automatically:
- Train the model on resting-state data
- Apply it to task data
- Generate error maps
- Create all visualizations

### 3. Run Individual Steps

Or run steps separately:

```bash
# Step 1: Train model
python scripts/01_train_model.py

# Step 2: Apply to task data
python scripts/02_apply_to_task.py

# Step 3: Create visualizations
python scripts/03_visualize_results.py
```

## Configuration

Edit `config.yaml` to customize:
- Data paths
- Model parameters
- Cross-validation settings
- Visualization options
- Error thresholds

## Output Files

All results are saved to `data/results/`:

### CSV Files
- `error_map_piop2_training.csv` - Training error rates per region
- `error_map_piop1.csv` - Task error rates per region
- `error_comparison_rest_vs_task.csv` - Comparison between conditions

### Visualizations
- `error_map_piop2.png` - Resting-state error map
- `error_map_piop1.png` - Task error map
- `comparison_rest_vs_task.png` - Rest vs task comparison
- `network_analysis.png` - Network-level analysis

## Usage as a Python Package

```python
from brain_pipeline import (
    Config, DataLoader, ConnectivityProcessor,
    BrainRegionClassifier, ModelEvaluator, Visualizer
)

# Load configuration
config = Config()

# Load data
data_loader = DataLoader(config)
df = data_loader.load_piop2()

# Process data
processor = ConnectivityProcessor()
region_list, region_to_idx, n_regions = processor.extract_regions(connection_columns)
X, y, subjects = processor.create_dataset(df, connection_columns)

# Train model
classifier = BrainRegionClassifier(config)
classifier.train(X, y)

# Evaluate
y_pred, y_proba = classifier.predict(X)
evaluator = ModelEvaluator(config, region_list)
error_map = evaluator.calculate_error_map(y, y_pred)

# Visualize
visualizer = Visualizer(config)
fig = visualizer.plot_error_map(error_map)
visualizer.save_figure(fig, 'my_error_map.png')
```

## Key Features

✅ **Modular Design** - Each component is independent and reusable  
✅ **Configurable** - Easy to adjust parameters via YAML config  
✅ **Type Hints** - Full type annotations for better IDE support  
✅ **Documentation** - Comprehensive docstrings  
✅ **Error Handling** - Robust error checking  
✅ **Reproducible** - Fixed random seeds for reproducibility  

## Methods

### Classification Approach
- **Model**: Multinomial Logistic Regression
- **Features**: Connectivity pattern for each region (excluding diagonal)
- **Labels**: Brain region identity (multi-class)
- **Validation**: Subject-wise cross-validation

### Error Map Interpretation
- **High error rate** → Region shows altered connectivity during task
- **Low error rate** → Region maintains stable connectivity patterns
- **Error increase** → Task-induced connectivity changes
