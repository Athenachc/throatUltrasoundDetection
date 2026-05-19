import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

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

# Define environments based on file naming convention
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

# Map ALL SAM2 configurations based on runs/train indexes matching the benchmark sheets
sam_map = {
    'train9':  'YOLOv8n+SAM2 (Hybrid Dataset)',
    'train10': 'YOLOv10n+SAM2 (Hybrid Dataset)',
    'train11': 'YOLOv11n+SAM2 (Hybrid Dataset)',
    'train3':  'YOLOv8n+SAM2 (In-house Dataset)',
    'train8':  'YOLOv10n+SAM2 (In-house Dataset)',
    'train5':  'YOLOv11n+SAM2 (In-house Dataset)'
}
df_sam_filtered = df_sam[df_sam['Model'].isin(sam_map.keys())].copy()
df_sam_filtered['Architecture'] = df_sam_filtered['Model'].map(sam_map)

# Combine datasets together for structural plotting
cols_to_keep = ['Video', 'Architecture', 'Environment', 'FPS', 'Mean Dice', 'Avg Latency (ms)']
df_combined = pd.concat([df_unet_filtered[cols_to_keep], df_sam_filtered[cols_to_keep]], ignore_index=True)

# Adjust anomalous FPS fields where tracking artifacts flag high processing rates on drops
df_combined.loc[df_combined['FPS'] > 50, 'FPS'] = 30.0

# ==========================================
# 2. DYNAMICALLY RENAME INDIVIDUAL VIDEOS
# ==========================================
controlled_vids = sorted(df_combined[df_combined['Environment'] == 'Controlled']['Video'].unique())
generalization_vids = sorted(df_combined[df_combined['Environment'] == 'Generalization']['Video'].unique())

video_rename_dict = {}
for idx, vid in enumerate(controlled_vids):
    video_rename_dict[vid] = f"In-house Video {idx + 1}"
for idx, vid in enumerate(generalization_vids):
    video_rename_dict[vid] = f"Web-sourced Video {idx + 1}"

df_combined['Video_Renamed'] = df_combined['Video'].map(video_rename_dict)

# Explicit layout tracking order along x-axis
video_axis_order = [f"In-house Video {i}" for i in range(1, 8)] + [f"Web-sourced Video {i}" for i in range(1, 7)]
df_combined['Video_Renamed'] = pd.Categorical(df_combined['Video_Renamed'], categories=video_axis_order, ordered=True)

# Map categories to positions to allow clean micro-jitter rendering
video_pos_map = {name: idx for idx, name in enumerate(video_axis_order)}
df_combined['Video_Pos'] = df_combined['Video_Renamed'].map(video_pos_map).astype(float)

# ==========================================
# 3. MASTER STYLING AND DICTIONARIES
# ==========================================
all_models = [
    'YOLOv8n+SAM2 (Hybrid Dataset)', 'YOLOv10n+SAM2 (Hybrid Dataset)', 'YOLOv11n+SAM2 (Hybrid Dataset)',
    'YOLOv8n+SAM2 (In-house Dataset)', 'YOLOv10n+SAM2 (In-house Dataset)', 'YOLOv11n+SAM2 (In-house Dataset)',
    'YOLOv8n+U-Net (Hybrid Dataset)', 'YOLOv8n+U-Net (In-house Dataset)', 'Plain U-Net (In-house Dataset)'
]

# Generate explicit color map using the standard tab10 palette
palette_colors = plt.get_cmap('tab10').colors
model_color_dict = {model: palette_colors[i % 10] for i, model in enumerate(all_models)}

sam2_models = all_models[0:6]
framework_comparison_models = [
    'YOLOv8n+SAM2 (Hybrid Dataset)',
    'YOLOv8n+U-Net (Hybrid Dataset)', 
    'YOLOv8n+U-Net (In-house Dataset)',
    'Plain U-Net (In-house Dataset)'
]

# Micro visual offsets to cleanly expose overlapping curves on identical nodes
jitter_map = {
    'YOLOv8n+SAM2 (Hybrid Dataset)': -0.08,
    'YOLOv10n+SAM2 (Hybrid Dataset)': 0.00,
    'YOLOv11n+SAM2 (Hybrid Dataset)': 0.08,
    'YOLOv8n+SAM2 (In-house Dataset)': -0.05,
    'YOLOv10n+SAM2 (In-house Dataset)': 0.00,
    'YOLOv11n+SAM2 (In-house Dataset)': 0.05,
    'YOLOv8n+U-Net (Hybrid Dataset)': -0.05,
    'YOLOv8n+U-Net (In-house Dataset)': 0.00,
    'Plain U-Net (In-house Dataset)': 0.05
}

df_combined['Jitter_Pos'] = df_combined['Video_Pos'] + df_combined['Architecture'].map(jitter_map).astype(float)

df_sam2_only = df_combined[df_combined['Architecture'].isin(sam2_models)].copy()
df_framework_only = df_combined[df_combined['Architecture'].isin(framework_comparison_models)].copy()

df_sam2_only['Architecture'] = pd.Categorical(df_sam2_only['Architecture'], categories=sam2_models, ordered=True)
df_framework_only['Architecture'] = pd.Categorical(df_framework_only['Architecture'], categories=framework_comparison_models, ordered=True)

sam2_marker_dict = {
    'YOLOv8n+SAM2 (Hybrid Dataset)': 'o', 'YOLOv10n+SAM2 (Hybrid Dataset)': 'X', 'YOLOv11n+SAM2 (Hybrid Dataset)': '^',
    'YOLOv8n+SAM2 (In-house Dataset)': 's', 'YOLOv10n+SAM2 (In-house Dataset)': 'P', 'YOLOv11n+SAM2 (In-house Dataset)': 'D'
}

framework_marker_dict = {
    'YOLOv8n+SAM2 (Hybrid Dataset)': 'o', 'YOLOv8n+U-Net (Hybrid Dataset)': 's',
    'YOLOv8n+U-Net (In-house Dataset)': '^', 'Plain U-Net (In-house Dataset)': 'D'
}

def add_domain_divider(max_y):
    plt.axvline(x=6.5, color='crimson', linestyle='--', linewidth=1.2, alpha=0.7)
    plt.text(6.3, max_y * 0.94, 'Controlled Domain', color='crimson', ha='right', fontsize=10, fontweight='bold')
    plt.text(6.7, max_y * 0.94, 'Generalization Domain', color='crimson', ha='left', fontsize=10, fontweight='bold')

def standardize_legend(ax, marker_size=7, line_width=1.0):
    """Enforces absolute fairness and item equality inside the legend box across backend variations."""
    leg = ax.get_legend()
    if leg:
        handles = getattr(leg, 'legend_handles', None) or getattr(leg, 'legendHandles', [])
        for handle in handles:
            if hasattr(handle, 'set_sizes'):
                handle.set_sizes([marker_size ** 2])
            if hasattr(handle, 'set_markersize'):
                handle.set_markersize(marker_size)
            if hasattr(handle, 'set_linewidth'):
                handle.set_linewidth(line_width)


# ==============================================================================
# PART A: MEAN DICE LINE GRAPHS
# ==============================================================================
# Graph 1: Mean Dice - All YOLO+SAM2 Models
fig, ax = plt.subplots(figsize=(13, 6.0))
for model in sam2_models:
    data = df_sam2_only[df_sam2_only['Architecture'] == model].sort_values('Video_Pos')
    ax.plot(data['Jitter_Pos'], data['Mean Dice'], label=model, linewidth=1.0, color=model_color_dict[model],
            marker=sam2_marker_dict[model], markersize=7, markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)
add_domain_divider(1.0)
plt.title('Video-by-Video Evaluation: Mean Dice Comparison (All YOLO+SAM2 Variants)', pad=12, fontweight='bold')
plt.xlabel('Dataset Identification Sequence', labelpad=10)
plt.ylabel('Mean Dice Coefficient')
plt.xticks(range(len(video_axis_order)), video_axis_order, rotation=30, ha='right')
plt.ylim(-0.05, 1.05)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
standardize_legend(ax)
plt.tight_layout()
plt.savefig('dice_line_sam2_variants.png', dpi=300)
plt.close()

# Graph 2: Mean Dice - Framework Comparison (4 Models)
fig, ax = plt.subplots(figsize=(13, 6.0))
for model in framework_comparison_models:
    data = df_framework_only[df_framework_only['Architecture'] == model].sort_values('Video_Pos')
    ax.plot(data['Jitter_Pos'], data['Mean Dice'], label=model, linewidth=1.0, color=model_color_dict[model],
            marker=framework_marker_dict[model], markersize=7, markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)
add_domain_divider(1.0)
plt.title('Video-by-Video Evaluation: Mean Dice Comparison (SAM2 vs. U-Net Frameworks)', pad=12, fontweight='bold')
plt.xlabel('Dataset Identification Sequence', labelpad=10)
plt.ylabel('Mean Dice Coefficient')
plt.xticks(range(len(video_axis_order)), video_axis_order, rotation=30, ha='right')
plt.ylim(-0.05, 1.05)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
standardize_legend(ax)
plt.tight_layout()
plt.savefig('dice_line_framework_comparison.png', dpi=300)
plt.close()


# ==============================================================================
# PART B: INFERENCE FPS LINE GRAPHS
# ==============================================================================
max_fps_val = df_combined['FPS'].max() + 3

# Graph 3: Inference FPS - All YOLO+SAM2 Models
fig, ax = plt.subplots(figsize=(13, 6.0))
for model in sam2_models:
    data = df_sam2_only[df_sam2_only['Architecture'] == model].sort_values('Video_Pos')
    ax.plot(data['Jitter_Pos'], data['FPS'], label=model, linewidth=1.0, color=model_color_dict[model],
            marker=sam2_marker_dict[model], markersize=7, markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)
add_domain_divider(max_fps_val)
plt.title('Video-by-Video Evaluation: Inference Throughput (All YOLO+SAM2 Variants)', pad=12, fontweight='bold')
plt.xlabel('Dataset Identification Sequence', labelpad=10)
plt.ylabel('Frames Per Second (FPS)')
plt.xticks(range(len(video_axis_order)), video_axis_order, rotation=30, ha='right')
plt.ylim(-1, max_fps_val)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
standardize_legend(ax)
plt.tight_layout()
plt.savefig('fps_line_sam2_variants.png', dpi=300)
plt.close()

# Graph 4: Inference FPS - Framework Comparison (4 Models)
fig, ax = plt.subplots(figsize=(13, 6.0))
for model in framework_comparison_models:
    data = df_framework_only[df_framework_only['Architecture'] == model].sort_values('Video_Pos')
    ax.plot(data['Jitter_Pos'], data['FPS'], label=model, linewidth=1.0, color=model_color_dict[model],
            marker=framework_marker_dict[model], markersize=7, markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)
add_domain_divider(max_fps_val)
plt.title('Video-by-Video Evaluation: Inference Throughput (SAM2 vs. U-Net Frameworks)', pad=12, fontweight='bold')
plt.xlabel('Dataset Identification Sequence', labelpad=10)
plt.ylabel('Frames Per Second (FPS)')
plt.xticks(range(len(video_axis_order)), video_axis_order, rotation=30, ha='right')
plt.ylim(-1, max_fps_val)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
standardize_legend(ax)
plt.tight_layout()
plt.savefig('fps_line_framework_comparison.png', dpi=300)
plt.close()


# ==============================================================================
# PART C: LATENCY VS ACCURACY SCATTER METRIC PLOTS
# ==============================================================================
grouped_sam2 = df_sam2_only.groupby(['Architecture', 'Environment'], observed=False)[['Mean Dice', 'Avg Latency (ms)']].mean().reset_index()
grouped_framework = df_framework_only.groupby(['Architecture', 'Environment'], observed=False)[['Mean Dice', 'Avg Latency (ms)']].mean().reset_index()

# Cast offsets to float explicitly to fix the categorical math error
grouped_sam2['Jitter_Lat'] = grouped_sam2['Avg Latency (ms)'] + grouped_sam2['Architecture'].map(jitter_map).astype(float) * 10
grouped_framework['Jitter_Lat'] = grouped_framework['Avg Latency (ms)'] + grouped_framework['Architecture'].map(jitter_map).astype(float) * 10

zone_patch = mpatches.Patch(facecolor='gray', alpha=0.15, edgecolor='none', label='High-Efficiency Zone (< 160 ms)')

# Graph 5: Scatter Matrix - All YOLO+SAM2 Models
fig, ax5 = plt.subplots(figsize=(11, 5.5))
ax5.axvspan(-20, 160, color='gray', alpha=0.15, zorder=1)

# Plot manually to maintain strict color/marker configuration and represent environment cleanly
for idx, row in grouped_sam2.iterrows():
    m_arch = row['Architecture']
    m_env = row['Environment']
    if pd.isna(m_arch): continue
    
    if m_env == 'Generalization':
        ax5.scatter(row['Jitter_Lat'], row['Mean Dice'], color=model_color_dict[m_arch], marker=sam2_marker_dict[m_arch],
                    facecolors='none', edgecolors=model_color_dict[m_arch], s=130, linewidths=2.0, zorder=3)
    else:
        ax5.scatter(row['Jitter_Lat'], row['Mean Dice'], color=model_color_dict[m_arch], marker=sam2_marker_dict[m_arch],
                    s=130, edgecolor='black', linewidths=0.6, zorder=3)

# Build unified academic legend card
leg_elements_5 = []
for model in sam2_models:
    leg_elements_5.append(mlines.Line2D([], [], color=model_color_dict[model], marker=sam2_marker_dict[model],
                                        linestyle='None', markersize=8, markeredgecolor='black', markeredgewidth=0.5, label=model))
leg_elements_5.append(mlines.Line2D([], [], color='none', linestyle='None', label='')) # Blank separation line
leg_elements_5.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=8, markeredgecolor='black', label='Controlled Domain (Solid)'))
leg_elements_5.append(mlines.Line2D([], [], color='none', marker='o', markerfacecolor='none', linestyle='None', markersize=8, markeredgecolor='gray', markeredgewidth=1.5, label='Generalization Domain (Hollow)'))
leg_elements_5.append(zone_patch)

plt.legend(handles=leg_elements_5, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.title('Latency vs. Accuracy Matrix: YOLO+SAM2 Optimization Space', pad=12, fontweight='bold')
plt.xlabel('Average Inference Latency (ms)')
plt.ylabel('Mean Dice Score')
plt.ylim(-0.05, 1.05)
plt.xlim(-20, grouped_sam2['Avg Latency (ms)'].max() + 50)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig('scatter_tradeoff_sam2_variants.png', dpi=300)
plt.close()


# Graph 6: Scatter Matrix - Framework Comparison (4 Models)
fig, ax6 = plt.subplots(figsize=(11, 5.5))
ax6.axvspan(-20, 160, color='gray', alpha=0.15, zorder=1)

# Plot manually to maintain strict color/marker configuration and represent environment cleanly
for idx, row in grouped_framework.iterrows():
    m_arch = row['Architecture']
    m_env = row['Environment']
    if pd.isna(m_arch): continue
    
    if m_env == 'Generalization':
        ax6.scatter(row['Jitter_Lat'], row['Mean Dice'], color=model_color_dict[m_arch], marker=framework_marker_dict[m_arch],
                    facecolors='none', edgecolors=model_color_dict[m_arch], s=130, linewidths=2.0, zorder=3)
    else:
        ax6.scatter(row['Jitter_Lat'], row['Mean Dice'], color=model_color_dict[m_arch], marker=framework_marker_dict[m_arch],
                    s=130, edgecolor='black', linewidths=0.6, zorder=3)

# Build unified academic legend card
leg_elements_6 = []
for model in framework_comparison_models:
    leg_elements_6.append(mlines.Line2D([], [], color=model_color_dict[model], marker=framework_marker_dict[model],
                                        linestyle='None', markersize=8, markeredgecolor='black', markeredgewidth=0.5, label=model))
leg_elements_6.append(mlines.Line2D([], [], color='none', linestyle='None', label='')) # Blank separation line
leg_elements_6.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=8, markeredgecolor='black', label='Controlled Domain (Solid)'))
leg_elements_6.append(mlines.Line2D([], [], color='none', marker='o', markerfacecolor='none', linestyle='None', markersize=8, markeredgecolor='gray', markeredgewidth=1.5, label='Generalization Domain (Hollow)'))
leg_elements_6.append(zone_patch)

plt.legend(handles=leg_elements_6, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.title('Latency vs. Accuracy Matrix: SAM2 vs. U-Net Pipeline Comparison', pad=12, fontweight='bold')
plt.xlabel('Average Inference Latency (ms)')
plt.ylabel('Mean Dice Score')
plt.ylim(-0.05, 1.05)
plt.xlim(-20, grouped_framework['Avg Latency (ms)'].max() + 50)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig('scatter_tradeoff_framework_comparison.png', dpi=300)
plt.close()

print("Structural adjustment complete. 6 completely balanced and legible graphs generated successfully.")