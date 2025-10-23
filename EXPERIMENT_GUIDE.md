# Experiment Guide: Running Notebooks

This guide explains how to run the various experiments in the Jupyter notebooks for the IT2TSK fuzzy inference system research.

## Directory Structure

```
notebooks/
├── experiment_1_mamdani_tsk_selection.ipynb    # FIS method comparison
├── experiment_2_clustering_algorithm.ipynb     # Clustering algorithm evaluation
├── experiment_3_mbk_means_exploration.ipynb    # MBK parameter tuning
├── experiment_4_gaussian_exploration.ipynb     # Gaussian MF parameter tuning
├── experiment_5_hyperparameter_tuning.ipynb    # GA optimization
├── experiment_6_performance_evaluation.ipynb   # Final performance analysis
├── playground_sensor_reading_plot.ipynb        # Sensor data visualization
└── data/                                       # Dataset files
    └── e-nose_dataset_12_beef_cuts.xlsx
```

## Quick Start

### Prerequisites

1. **Python Environment**: Ensure Python 3.10+ is installed
2. **Dependencies**: Install required packages from `requirements.txt`
3. **Jupyter**: Have Jupyter Notebook or VS Code with Jupyter extension

### Setup Steps

> **Terminal Commands**: All commands starting with `$` or shown in code blocks below should be run in your terminal/command prompt. Open Terminal (macOS/Linux) or Command Prompt/PowerShell (Windows) to execute these commands.

```bash
# 1. Navigate to experiment directory
$ cd /path/to/my_experiment

# 2. Install dependencies
$ pip install -r requirements.txt

# 3. Launch Jupyter (if using standalone Jupyter)
$ jupyter notebook notebooks/

# 4. Or open in VS Code
$ code notebooks/
```

> **Important**: When running terminal commands (like `bin/mamdani_tsk_selection`), Excel result files are created in the **project root directory**. To organize results and prevent overwriting previous experiments, manually move these files to `notebooks/experiments/` after each run.

## Experiment Workflow

### Experiment 1: FIS Method Selection

**File**: `experiment_1_mamdani_tsk_selection.ipynb`

**Purpose**: Compare different fuzzy inference system types

- Mamdani vs TSK vs IT2-TSK
- Different membership function types
- Various clustering methods

**How to Run**:

1. Open the notebook
2. Execute cells sequentially from top to bottom
3. **Key Cell**: Cell with `bin/mamdani_tsk_selection` command
4. **Output**: `Mamdani_TSK_Selection.xlsx` (created in project root)
5. **File Management**: `$ mv Mamdani_TSK_Selection.xlsx notebooks/experiments/`
6. **Time**: ~5-10 minutes

**Key Parameters**:

```python
# Modify these in the notebook
clustering_methods = [ClusteringMethod.MBKMEANS, ClusteringMethod.KMEANS]
mf_types = [MFType1.TRIANGULAR, MFType1.GAUSSIAN, MFType2.GAUSSIAN]
```

### Experiment 2: Clustering Algorithm Evaluation

**File**: `experiment_2_clustering_algorithm.ipynb`

**Purpose**: Evaluate different clustering algorithms

- K-Means vs Mini-Batch K-Means vs Fuzzy C-Means
- Performance comparison
- Computational efficiency analysis

**How to Run**:

1. Execute `bin/clustering_performance` command
2. Analyze results in generated Excel file
3. **Output**: `Clustering_Performance.xlsx`

### Experiment 3: MBK Parameter Exploration

**File**: `experiment_3_mbk_means_exploration.ipynb`

**Purpose**: Optimize Mini-Batch K-Means parameters

- `batch_size`: [256, 512, 1024]
- `tol`: [1e-5, 1e-4, 1e-3]
- `max_no_improvement`: [5, 10, 20]

**How to Run**:

1. Execute `bin/it2tsk_mbk_exploration`
2. Monitor progress in logs
3. **Output**: `IT2TSK_MBKMeans_Exploration.xlsx`
4. **Time**: ~30-45 minutes (324 combinations × 12 worksheets)

**Monitoring Progress**:

```bash
# Watch log file in real-time
$ tail -f logs/it2tsk_mbk_exploration.log
```

### Experiment 4: Gaussian MF Parameter Exploration

**File**: `experiment_4_gaussian_exploration.ipynb`

**Purpose**: Optimize Gaussian membership function parameters

- `uncertainty_factor`: [0.01, 0.30] step 0.02
- `min_std_ratio`: [0.01, 0.18] step 0.02

**How to Run**:

1. Execute `bin/it2tsk_gaussian_exploration`
2. **Output**: `IT2TSK_Gaussian_Exploration.xlsx`
3. **Time**: ~15-20 minutes

### Experiment 5: Hyperparameter Tuning (GA)

**File**: `experiment_5_hyperparameter_tuning.ipynb`

**Purpose**: Genetic Algorithm optimization

- Uses results from Experiments 3 & 4
- 10 GA runs per worksheet
- Population size: 6, Generations: 25

**How to Run**:

1. **Prerequisite**: Complete Experiments 3 & 4 first
2. Execute `bin/it2tsk_hyperparameter_tuning`
3. **Output**: `IT2TSK_Hyperparameter_Tuning.xlsx`
4. **Time**: ~45-60 minutes

**GA Parameters**:

```python
sol_per_pop = 6           # Population size
num_generations = 25      # Maximum generations
num_parents_mating = 3    # Parents for crossover
stop_criteria = "saturate_5"  # Early stopping
```

### Experiment 6: Performance Evaluation

**File**: `experiment_6_performance_evaluation.ipynb`

**Purpose**: Final comparison with other ML methods

- IT2TSK vs TSK vs MLP vs SVR vs RF vs KNN
- Statistical analysis
- Performance metrics: R², MSE, MAE

**How to Run**:

1. Execute `bin/methods_comparison`
2. **Output**: `Method_Comparisons.xlsx`
3. **Time**: ~2-3 minutes

## Playground Notebooks

### Sensor Reading Visualization

**File**: `playground_sensor_reading_plot.ipynb`

**Purpose**: Visualize sensor data and clustering

- Sensor value normalization
- Cluster visualization
- Membership function plotting

**Key Features**:

- **Interactive parameters**: Change `worksheet` and `sensor` variables
- **Color consistency**: Fixed cluster-centroid color mapping
- **Statistics**: Detailed cluster analysis

**How to Customize**:

```python
# Change these parameters in Cell 2
worksheet = WorkSheet.DS8.value  # Choose worksheet (DS1-DS12)
sensor = 'MQ4'                   # Choose sensor (MQ2, MQ3, MQ4, etc.)
```

## Data Flow

```
Raw Data (e-nose_dataset_12_beef_cuts.xlsx)
    ↓
Experiment 1: FIS Selection
    ↓
Experiment 2: Clustering Evaluation
    ↓
Experiment 3: MBK Parameter Optimization
    ↓
Experiment 4: Gaussian MF Optimization
    ↓
Experiment 5: GA Hyperparameter Tuning
    ↓
Experiment 6: Final Performance Evaluation
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

```python
# If you get module not found errors
import sys
sys.path.append('/path/to/my_experiment')
from notebook_resolver import *
```

#### 2. Memory Issues

- **Symptom**: Kernel crashes during large experiments
- **Solution**: Reduce population size in GA experiments

```python
sol_per_pop = 4  # Reduce from 6 to 4
```

#### 3. Long Execution Times

- **MBK Exploration**: Normal 30-45 minutes
- **GA Tuning**: Normal 45-60 minutes
- **Monitor progress**: Use log files in `logs/` directory

#### 4. Color Mismatch in Plots

- **Issue**: Clusters and centroids have different colors
- **Solution**: Use the updated plotting code with label remapping

### Performance Optimization

#### For Faster Experiments:

```python
# Reduce parameter ranges
uncertainty_factors = np.arange(0.01, 0.10, 0.02)  # Smaller range
min_std_ratios = np.arange(0.01, 0.10, 0.02)       # Smaller range

# Reduce GA parameters
num_generations = 15        # From 25 to 15
sol_per_pop = 4            # From 6 to 4
```

## Expected Outputs

### Excel Files Generated:

> **File Location**: All Excel files are created in the **project root directory** when running terminal commands. To keep results organized and avoid overwriting previous experiments, **manually move** these files to `notebooks/experiments/` directory after each experiment.

- `Mamdani_TSK_Selection.xlsx`: FIS comparison results
- `IT2TSK_MBKMeans_Exploration.xlsx`: MBK parameter optimization
- `IT2TSK_Gaussian_Exploration.xlsx`: Gaussian MF optimization
- `IT2TSK_Hyperparameter_Tuning.xlsx`: GA optimization results
- `Method_Comparisons.xlsx`: Final performance comparison

**Recommended File Management**:

```bash
# After running an experiment, move the generated file:
$ mv Mamdani_TSK_Selection.xlsx notebooks/experiments/
$ mv IT2TSK_MBKMeans_Exploration.xlsx notebooks/experiments/
# ... and so on for other generated files
```

### Log Files:

- `logs/it2tsk_mbk_exploration.log`: MBK exploration progress
- `logs/it2tsk_hyperparameter_tuning.log`: GA optimization progress

## Key Success Metrics

### Target Performance:

- **R² Score**: > 0.95 for beef freshness prediction
- **MSE**: < 0.1 for normalized TVC values
- **Computational Time**: < 60 seconds per prediction

### Validation Criteria:

1. **Consistency**: Results reproducible across runs
2. **Convergence**: GA algorithms reach stable solutions
3. **Comparison**: IT2TSK outperforms traditional methods

## Notes

### Best Practices:

1. **Sequential Execution**: Run experiments in order (1-6)
2. **Save Checkpoints**: Export intermediate results to Excel
3. **Monitor Logs**: Watch progress for long-running experiments
4. **Parameter Documentation**: Record parameter changes in notebook cells

### Research Tips:

- **Experiment with Parameters**: Try different ranges for your dataset
- **Visualize Results**: Use playground notebooks to understand data patterns
- **Compare Methods**: Always include baseline comparisons
- **Statistical Validation**: Use multiple runs for robust results

---

For questions or issues, refer to the individual notebook documentation or check the `logs/` directory for detailed execution information.
