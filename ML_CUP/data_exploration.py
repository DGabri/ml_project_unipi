"""
Data Exploration and Visualization for ML Cup Dataset
This script generates comprehensive visualizations including:
1. Correlation matrix (features vs features)
2. Correlation matrix (features vs targets)
3. Distribution plots for all input variables
4. Pairwise scatter plots for highly correlated features
5. Statistical summary
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import os

from ml_cup_data_loader import load_training_set

# Create output directory
os.makedirs("./results/data_exploration", exist_ok=True)

# Load data
print("Loading training data...")
train_df = load_training_set()

# Separate features and targets
X = train_df.iloc[:, :-4]
y = train_df.iloc[:, -4:]

# Rename columns for clarity
X.columns = [f'X{i+1}' for i in range(X.shape[1])]
y.columns = [f'Y{i+1}' for i in range(y.shape[1])]

print(f"Dataset shape: {train_df.shape}")
print(f"Features: {X.shape[1]}, Targets: {y.shape[1]}, Samples: {X.shape[0]}")

############################################
# 1. CORRELATION MATRIX - FEATURES
############################################

print("\nGenerating feature correlation matrix...")

# Calculate correlation matrix
corr_matrix = X.corr()

# Create figure
fig, ax = plt.subplots(figsize=(14, 12))

# Plot heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Mask upper triangle
sns.heatmap(corr_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f', 
            cmap='RdBu_r', 
            center=0,
            vmin=-1, 
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
            ax=ax)

ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print("✓ Feature correlation matrix saved")

############################################
# 2. CORRELATION MATRIX - FEATURES vs TARGETS
############################################

print("Generating feature-target correlation matrix...")

# Combine for correlation calculation
combined_df = pd.concat([X, y], axis=1)
corr_full = combined_df.corr()

# Extract feature-target correlations
feature_target_corr = corr_full.iloc[:X.shape[1], X.shape[1]:]

# Create figure
fig, ax = plt.subplots(figsize=(8, 14))

sns.heatmap(feature_target_corr, 
            annot=True, 
            fmt='.2f', 
            cmap='RdBu_r', 
            center=0,
            vmin=-1, 
            vmax=1,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
            ax=ax)

ax.set_title('Feature-Target Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Target Variables', fontsize=12, fontweight='bold')
ax.set_ylabel('Input Features', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print("✓ Feature-target correlation matrix saved")

############################################
# 3. DISTRIBUTION PLOTS FOR INPUT VARIABLES
############################################

print("Generating distribution plots for input variables...")

# Calculate grid dimensions
n_features = X.shape[1]
n_cols = 4
n_rows = int(np.ceil(n_features / n_cols))

# Create figure with subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
axes = axes.flatten()

# Define colors for each plot
colors = plt.cm.Set3(np.linspace(0, 1, n_features))

# Plot each feature
for idx, col in enumerate(X.columns):
    ax = axes[idx]
    data = X[col].values

    # Create histogram with KDE overlay
    ax.hist(data, bins=30, density=True, alpha=0.7, color=colors[idx], edgecolor='black', linewidth=0.5)

    # Add KDE curve
    kde_xs = np.linspace(data.min(), data.max(), 200)
    kde = stats.gaussian_kde(data)
    ax.plot(kde_xs, kde(kde_xs), color='black', linewidth=2, alpha=0.8)

    # Styling
    ax.set_xlabel('Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Input {idx+1}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

# Hide unused subplots
for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Distribution of Input Features', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()
print("✓ Input distribution plots saved")
