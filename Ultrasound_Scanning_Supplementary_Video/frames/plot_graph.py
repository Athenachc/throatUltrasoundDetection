import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

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
path_unet_summary = '/home/athena/Downloads/comparison/video_summary.csv'
path_sam_summary = '/home/athena/Downloads/comparison/hybrid_benchmark_results_old.csv'

# Load both datasets
df_unet = pd.read_csv(path_unet_summary)
df_sam = pd.read_csv(path_sam_summary)

# Define environments based on your file naming convention
df_unet['Environment'] = df_unet['Video'].apply(lambda x: 'Generalization' if 'video' in x else 'Controlled')
df_sam['Environment'] = df_sam['Video'].apply(lambda x: 'Generalization' if 'video' in x else 'Controlled')

# Map the 3 architectures from the U-Net summary file
unet_map = {
    'plain_unet': 'Plain U-Net (In-house Dataset)',
    'train3': 'YOLOv8n+U-Net (In-house Dataset)',
    'train9': 'YOLOv8n+U-Net (Hybrid Dataset)'
}
df_unet_filtered = df_unet[df_unet['Model'].isin(unet_map.keys())].copy()
df_unet_filtered['Architecture'] = df_unet_filtered['Model'].map(unet_map)

# Map the SAM2 architecture from the old benchmark file (where train9 = SAM2)
df_sam_filtered = df_sam[df_sam['Model'] == 'train9'].copy()
df_sam_filtered['Architecture'] = 'YOLOv8n+SAM2 (Hybrid Dataset)'

# Combine the dataframes together for plotting (Added 'Avg Latency (ms)')
cols_to_keep = ['Video', 'Architecture', 'Environment', 'FPS', 'Mean Dice', 'Avg Latency (ms)']
df_combined = pd.concat([df_unet_filtered[cols_to_keep], df_sam_filtered[cols_to_keep]], ignore_index=True)

# Calculate the averages across ALL videos per Architecture and Environment
grouped_df = df_combined.groupby(['Architecture', 'Environment'])[['Mean Dice', 'FPS', 'Avg Latency (ms)']].mean().reset_index()


# ==========================================
# GRAPH 2: LATENCY VS ACCURACY PARETO (SCATTER)
# ==========================================
plt.figure(figsize=(10, 5.0))

# Enhanced background shading (alpha=0.15) for high-efficiency zone visibility
plt.axvspan(-20, 160, color='gray', alpha=0.15, zorder=1)

# Plot the scatter using 'Architecture' for the hue parameter
ax = sns.scatterplot(
    data=grouped_df,
    x='Avg Latency (ms)',
    y='Mean Dice',
    hue='Architecture',
    style='Environment',
    s=250, 
    palette=['#4C72B0', '#DD8452', '#2E8B57', '#9370DB'], 
    edgecolor='black',
    linewidth=1.2,
    zorder=3
)

# Chart labels and configurations
plt.title('Latency vs. Accuracy Trade-off Matrix', pad=15, fontweight='bold')
plt.xlabel(r'Average Inference Latency (ms)')
plt.ylabel(r'Segmentation Accuracy (Mean Dice Score)')

# Axis limits: set y-max tightly to 0.9 to flatten the layout height
plt.ylim(-0.05, 0.90)
plt.xlim(-20, max(grouped_df['Avg Latency (ms)']) + 120) 
plt.grid(True, linestyle="--", alpha=0.4, zorder=1)

# --------------------------------------------------
# CUSTOM LEGEND PARSING WITH AN INSERTED BLANK ROW
# --------------------------------------------------
handles, labels = ax.get_legend_handles_labels()

# Dynamically find the exact index where the 'Environment' subheader section starts
try:
    env_index = labels.index('Environment')
except ValueError:
    env_index = len(labels) // 2 

# Clean slicing based on the dynamic position of Environment
architecture_handles = handles[:env_index]
architecture_labels = labels[:env_index]

environment_handles = handles[env_index:]
environment_labels = labels[env_index:]

# Create a concrete patch matching the axvspan to represent the shaded zone
zone_handle = mpatches.Patch(color='gray', alpha=0.15, edgecolor='none')
zone_label = "High-Efficiency Zone (< 160 ms)"

# Create a blank structural patch proxy to act as a clean spacer line
blank_handle = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False)
blank_label = ""

# Reassemble components: Models -> Spacer -> Environments -> Zone Handle (at the absolute bottom)
new_handles = architecture_handles + [blank_handle] + environment_handles + [zone_handle]
new_labels = architecture_labels + [blank_label] + environment_labels + [zone_label]

# Render legend outside the plot frame
plt.legend(
    handles=new_handles,
    labels=new_labels,
    title='Architectural Pipeline & Training Regimes', 
    bbox_to_anchor=(1.02, 1), 
    loc='upper left', 
    frameon=True, 
    shadow=False
)

plt.tight_layout()
plt.savefig('latency_vs_accuracy_scatter.png', dpi=300)
plt.close()

print("Publication-ready compact scatter plot with shaded zone item fixed to the bottom of the legend.")