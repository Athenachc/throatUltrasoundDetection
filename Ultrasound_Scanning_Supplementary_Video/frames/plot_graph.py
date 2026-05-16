import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set formal academic plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

# ==========================================
# 1. READ AND PREPROCESS DATA FROM CSVS
# ==========================================
# Update this path if hybrid_benchmark_results_old.csv is in a different folder
path_unet_summary = '/home/athena/Downloads/comparison/video_summary.csv'
path_sam_summary = '/home/athena/Downloads/comparison/hybrid_benchmark_results_old.csv'

# Load both datasets
df_unet = pd.read_csv(path_unet_summary)
df_sam = pd.read_csv(path_sam_summary)

# Define environments based on your file naming convention
df_unet['Environment'] = df_unet['Video'].apply(lambda x: 'Generalization' if 'video' in x else 'Controlled')
df_sam['Environment'] = df_sam['Video'].apply(lambda x: 'Generalization' if 'video' in x else 'Controlled')

# Map the 3 methodologies from the U-Net summary file
unet_map = {
    'plain_unet': 'Plain U-Net (In-house)',
    'train3': 'YOLOv8n+U-Net (In-house)',
    'train9': 'YOLOv8n+U-Net (Hybrid)'
}
df_unet_filtered = df_unet[df_unet['Model'].isin(unet_map.keys())].copy()
df_unet_filtered['Methodology'] = df_unet_filtered['Model'].map(unet_map)

# Map the SAM2 methodology from the old benchmark file (where train9 = SAM2)
df_sam_filtered = df_sam[df_sam['Model'] == 'train9'].copy()
df_sam_filtered['Methodology'] = 'YOLOv8n+SAM2 (Hybrid)'

# Combine the dataframes together for plotting
cols_to_keep = ['Video', 'Methodology', 'Environment', 'FPS', 'Mean Dice']
df_combined = pd.concat([df_unet_filtered[cols_to_keep], df_sam_filtered[cols_to_keep]], ignore_index=True)

# Calculate the averages across ALL videos per Model and Environment
grouped_df = df_combined.groupby(['Methodology', 'Environment'])[['Mean Dice', 'FPS']].mean().reset_index()

# ==========================================
# GRAPH 1: GENERALIZATION GAP (BAR CHART)
# ==========================================
plt.figure(figsize=(9, 5.5))

sns.barplot(
    data=grouped_df, 
    x='Methodology', 
    y='Mean Dice', 
    hue='Environment', 
    palette=['#4C72B0', '#C44E52'], # Academic blue and muted red
    edgecolor='black',
    linewidth=1
)

plt.title('Anatomical Generalization Gap Across Diverse Video Domains', pad=15, fontweight='bold')
plt.ylabel('Mean Dice Similarity Coefficient (DSC)')
plt.xlabel('Evaluated Segmentation Architectures')
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.legend(title='Testing Domain', loc='upper right')
plt.tight_layout()

plt.savefig('generalization_gap_analysis.png', dpi=300)
plt.close()

# ==========================================
# GRAPH 2: SPEED VS ACCURACY PARETO (SCATTER)
# ==========================================
plt.figure(figsize=(9, 5.5))

# Plot the scatter using the aggregated data across all videos
sns.scatterplot(
    data=grouped_df,
    x='FPS',
    y='Mean Dice',
    hue='Methodology',
    style='Environment',
    s=250, # Marker size
    palette=['#4C72B0', '#DD8452', '#2E8B57', '#9370DB'], # Distinct colors
    edgecolor='black',
    linewidth=1.2
)

plt.title('Speed-Accuracy Trade-off (Averaged Across All Videos)', pad=15, fontweight='bold')
plt.xlabel('Average Inference Speed (FPS)')
plt.ylabel('Average Mean Dice Score')
plt.ylim(-0.05, 1.05)
plt.xlim(0, max(grouped_df['FPS']) + 2) # Dynamically pad the x-axis
plt.grid(True, linestyle="--", alpha=0.5)

# Place legend outside to avoid blocking data points
plt.legend(title='Methodology & Domain', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

plt.savefig('speed_accuracy_tradeoff.png', dpi=300)
plt.close()

print("Publication-ready figures generated directly from CSVs: 'generalization_gap_analysis.png' and 'speed_accuracy_tradeoff.png'")