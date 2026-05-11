import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('/home/athena/Downloads/hybrid_benchmark_results_old.csv')

# 1. Define Model Mappings based on Table labeling
# Architecture: YOLOvXn+SAM2 | Training Dataset: Hybrid or In-house
model_map = {
    'train9': 'YOLOv8n+SAM2 (Hybrid)',
    'train10': 'YOLOv10n+SAM2 (Hybrid)',
    'train11': 'YOLOv11n+SAM2 (Hybrid)',
    'train3': 'YOLOv8n+SAM2 (In-house)',
    'train8': 'YOLOv10n+SAM2 (In-house)',
    'train5': 'YOLOv11n+SAM2 (In-house)'
}
df['Model_Name'] = df['Model'].map(model_map)

# Defining Training Dataset names to match Table columns exactly
df['Training Dataset'] = df['Model_Name'].apply(lambda x: 'Hybrid' if 'Hybrid' in x else 'In-house')

# Extracting Architecture name (e.g., YOLOv8n+SAM2) for the X-axis
df['Architecture'] = df['Model_Name'].apply(lambda x: x.split(' ')[0])

# 2. Categorize Videos (Controlled vs Generalization)
df['Environment'] = df['Video'].apply(lambda x: 'Generalization' if 'video' in x else 'Controlled')

sns.set_style("whitegrid")

# --- Plot 1: Generalization Gap Bar Chart ---
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Architecture', y='Mean Dice', hue='Training Dataset', palette='muted', errorbar=None)
plt.title('Performance Robustness: Hybrid vs. In-house Training', fontsize=14, pad=15)
plt.ylabel('Mean Dice Score', fontsize=12)
plt.xlabel('Architecture (Detection + Segmentation)', fontsize=12)
plt.ylim(0, 1.0)
plt.legend(title='Training Dataset', loc='upper right')
plt.savefig('generalization_gap_v2.png', dpi=300, bbox_inches='tight')

# --- Plot 2: Speed vs. Accuracy Trade-off ---
plt.figure(figsize=(10, 6))
# Using Controlled data for fair speed comparison (avoids nominal FPS spikes)
ctrl_df = df[df['Environment'] == 'Controlled']
sns.scatterplot(data=ctrl_df, x='FPS', y='Mean Dice', hue='Training Dataset', style='Architecture', s=200, palette='viridis')
plt.title('Speed-Accuracy Pareto Front (Controlled Env)', fontsize=14)
plt.xlabel('Inference Speed (FPS)', fontsize=12)
plt.ylabel('Mean Dice Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('speed_accuracy_tradeoff_v2.png', dpi=300, bbox_inches='tight')

# --- Plot 3: Stability Boxplot ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Architecture', y='Mean Dice', hue='Training Dataset', palette='Set2')
plt.title('Inference Stability Across All Video Sequences', fontsize=14)
plt.ylabel('Mean Dice Score', fontsize=12)
plt.xlabel('Architecture')
plt.tight_layout()
plt.savefig('performance_stability_v2.png', dpi=300)

print("Graphs generated successfully: generalization_gap.png, speed_accuracy_tradeoff.png, performance_stability.png")