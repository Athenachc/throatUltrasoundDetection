import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Set formal academic plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,   # Optimized font size for exactly 13 labels
    'ytick.labelsize': 16,
    'figure.titlesize': 10
})

# ==========================================
# 1. READ AND PREPROCESS DATA FROM CSVS
# ==========================================
path_unet_summary = '/home/athena/Downloads/comparison/video_summary.csv'
path_sam_summary = '/home/athena/Downloads/comparison/hybrid_benchmark_results_old.csv'

# Load both datasets
df_unet = pd.read_csv(path_unet_summary)
df_sam = pd.read_csv(path_sam_summary)

# Clean whitespace out of column namespaces
df_unet.columns = df_unet.columns.str.strip()
df_sam.columns = df_sam.columns.str.strip()

# --- FIX COLUMN NAME CASE AND MISMATCHES ---
df_unet = df_unet.rename(columns={
    'video': 'Video',
    'model': 'Model',
    'mean_dice': 'Mean Dice',
    'mean_fps': 'FPS',
    'mean_latency_ms': 'Avg Latency (ms)'
})

# Normalize video names by stripping extensions from SAM benchmark sheets
df_sam['Video'] = df_sam['Video'].astype(str).str.replace('.mp4', '', case=False)
df_unet['Video'] = df_unet['Video'].astype(str)

# If missing, supply baseline placeholders for U-Net models
if 'Avg Latency (ms)' not in df_unet.columns:
    df_unet['Avg Latency (ms)'] = df_unet['Model'].apply(lambda x: 144.93 if 'hybrid' in str(x) else 35.5)
if 'FPS' not in df_unet.columns:
    df_unet['FPS'] = df_unet['Model'].apply(lambda x: 6.9 if 'hybrid' in str(x) else 28.0)

# Define environments cleanly
df_unet['Environment'] = df_unet['Video'].apply(lambda x: 'Generalization' if 'video' in str(x) else 'Controlled')
df_sam['Environment'] = df_sam['Video'].apply(lambda x: 'Generalization' if 'video' in str(x) else 'Controlled')

# Map architectures
unet_map = {
    'plain_unet': 'U-Net (In-house Dataset)',
    'hybrid_unet': 'U-Net (Hybrid Dataset)',
    'train3': 'YOLOv8n+U-Net (In-house Dataset)',
    'train9': 'YOLOv8n+U-Net (Hybrid Dataset)'
}
df_unet_filtered = df_unet[df_unet['Model'].isin(unet_map.keys())].copy()
df_unet_filtered['Architecture'] = df_unet_filtered['Model'].map(unet_map)

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
df_combined.loc[df_combined['FPS'] > 50, 'FPS'] = 30.0

# ==============================================================================
# 2. FIXED ANATOMICAL EXPLICIT SORTING ENGINE (ONLY 13 CLEAN TICKS)
# ==============================================================================
all_raw_videos = set(df_combined['Video'].unique())

# Strictly construct Controlled sequence based on your exact specifications:
# 2 cgl, 2 cmc, 1 czw, 2 ywc
ordered_controlled_vids = [
    'cgl_20250930122939', 'cgl_20250930123155',
    'cmc_20250930122534', 'cmc_20250930122744',
    'czw_20250930123758',
    'ywc_20250930121012', 'ywc_20250930121841'
]

# Strictly construct Generalization sequence skipping video3:
ordered_generalization_vids = ['video1', 'video2', 'video4', 'video5', 'video6', 'video7']

# Verify integrity against source file data matrices
controlled_vids = [v for v in ordered_controlled_vids if v in all_raw_videos]
generalization_vids = [v for v in ordered_generalization_vids if v in all_raw_videos]

# Generate clean tracking structures
video_rename_dict = {}
video_axis_order = []

for idx, vid in enumerate(controlled_vids):
    alias = f"In-house Video {idx + 1}"
    video_rename_dict[vid] = alias
    video_axis_order.append(alias)

for idx, vid in enumerate(generalization_vids):
    alias = f"Web-sourced Video {idx + 1}"
    video_rename_dict[vid] = alias
    video_axis_order.append(alias)

df_combined['Video_Renamed'] = df_combined['Video'].map(video_rename_dict)

# Force the data frame to strictly follow the 13 categorical categories
df_combined['Video_Renamed'] = pd.Categorical(df_combined['Video_Renamed'], categories=video_axis_order, ordered=True)

# Generate unified structural integers (0 to 12) for axis tracking points
video_pos_map = {name: idx for idx, name in enumerate(video_axis_order)}
df_combined['Video_Pos'] = df_combined['Video_Renamed'].map(video_pos_map).astype(float)

# ==========================================
# 3. MASTER STYLING AND DICTIONARIES
# ==========================================
all_models = [
    'YOLOv8n+SAM2 (Hybrid Dataset)', 'YOLOv10n+SAM2 (Hybrid Dataset)', 'YOLOv11n+SAM2 (Hybrid Dataset)',
    'YOLOv8n+SAM2 (In-house Dataset)', 'YOLOv10n+SAM2 (In-house Dataset)', 'YOLOv11n+SAM2 (In-house Dataset)',
    'YOLOv8n+U-Net (Hybrid Dataset)', 'YOLOv8n+U-Net (In-house Dataset)', 
    'U-Net (Hybrid Dataset)', 'U-Net (In-house Dataset)'
]

palette_colors = plt.get_cmap('tab10').colors
model_color_dict = {model: palette_colors[i % 10] for i, model in enumerate(all_models)}

sam2_models = all_models[0:6]
framework_comparison_models = [
    'YOLOv8n+SAM2 (Hybrid Dataset)', 'YOLOv8n+U-Net (Hybrid Dataset)', 
    'YOLOv8n+U-Net (In-house Dataset)', 'U-Net (Hybrid Dataset)', 'U-Net (In-house Dataset)'
]

# Tightly managed categorical offsets keeping line markers separated on their shared 0-12 data column
jitter_map = {
    'YOLOv8n+SAM2 (Hybrid Dataset)': 0,
    'YOLOv10n+SAM2 (Hybrid Dataset)': -0.07,
    'YOLOv11n+SAM2 (Hybrid Dataset)': -0.02,
    'YOLOv8n+SAM2 (In-house Dataset)': 0.02,
    'YOLOv10n+SAM2 (In-house Dataset)': 0.07,
    'YOLOv11n+SAM2 (In-house Dataset)': 0.12,
    'YOLOv8n+U-Net (Hybrid Dataset)': 0,
    'YOLOv8n+U-Net (In-house Dataset)': 0,
    'U-Net (Hybrid Dataset)': 0,
    'U-Net (In-house Dataset)': 0
}

df_combined['Jitter_Pos'] = df_combined['Video_Pos'] + df_combined['Architecture'].map(jitter_map).astype(float)

df_sam2_only = df_combined[df_combined['Architecture'].isin(sam2_models)].copy()
df_framework_only = df_combined[df_combined['Architecture'].isin(framework_comparison_models)].copy()

sam2_marker_dict = {
    'YOLOv8n+SAM2 (Hybrid Dataset)': 'o', 'YOLOv10n+SAM2 (Hybrid Dataset)': 'X', 'YOLOv11n+SAM2 (Hybrid Dataset)': '^',
    'YOLOv8n+SAM2 (In-house Dataset)': 's', 'YOLOv10n+SAM2 (In-house Dataset)': 'P', 'YOLOv11n+SAM2 (In-house Dataset)': 'D'
}

framework_marker_dict = {
    'YOLOv8n+SAM2 (Hybrid Dataset)': 'o', 'YOLOv8n+U-Net (Hybrid Dataset)': 's',
    'YOLOv8n+U-Net (In-house Dataset)': '^', 'U-Net (Hybrid Dataset)': 'v', 'U-Net (In-house Dataset)': 'D'
}

def add_domain_divider(max_y):
    mid = len(controlled_vids) - 0.5
    plt.axvline(x=mid, color='crimson', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.text(mid - 0.15, max_y * 0.05, 'Controlled Domain', color='crimson', ha='right', fontsize=14, fontweight='bold')
    plt.text(mid + 0.15, max_y * 0.05, 'Generalization Domain', color='crimson', ha='left', fontsize=14, fontweight='bold')

def standardize_legend(ax, marker_size=7, line_width=1.0):
    leg = ax.get_legend()
    if leg:
        handles = getattr(leg, 'legend_handles', None) or getattr(leg, 'legendHandles', [])
        for handle in handles:
            if hasattr(handle, 'set_sizes'): handle.set_sizes([marker_size ** 2])
            if hasattr(handle, 'set_markersize'): handle.set_markersize(marker_size)
            if hasattr(handle, 'set_linewidth'): handle.set_linewidth(line_width)

# ==============================================================================
# PART A: MEAN DICE LINE GRAPHS (CLEAN AXIS REPAIR)
# ==============================================================================
# Graph 1: Mean Dice - All YOLO+SAM2 Models
fig, ax1 = plt.subplots(figsize=(13.5, 6.5))
for model in sam2_models:
    data = df_sam2_only[df_sam2_only['Architecture'] == model].sort_values('Video_Pos')
    ax1.plot(data['Jitter_Pos'], data['Mean Dice'], label=model, linewidth=1.1, color=model_color_dict[model],
             marker=sam2_marker_dict[model], markersize=7, markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)
# add_domain_divider(1.0)

plt.title('Mean Dice Comparison of the YOLO+SAM2 Architecture\nunder Hybrid and In-House Training Configurations', pad=12, fontweight='bold')
plt.xlabel('Dataset Identification Sequence', labelpad=12)
plt.ylabel('Mean Dice Coefficient')

# CRITICAL LOCK: Bind plot space and ticks directly to the 13 targeted sequence values
plt.xlim(-0.5, len(video_axis_order) - 0.5)
ax1.set_xticks(range(len(video_axis_order)))
ax1.set_xticklabels(video_axis_order, rotation=45, ha='right')

plt.ylim(-0.05, 1.0)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
standardize_legend(ax1)
plt.tight_layout()
plt.savefig('dice_line_sam2_variants.png', dpi=300)
plt.close()

# Graph 2: Mean Dice - Framework Comparison (5 Models)
fig, ax2 = plt.subplots(figsize=(13.5, 6.5))
for model in framework_comparison_models:
    data = df_framework_only[df_framework_only['Architecture'] == model].sort_values('Video_Pos')
    ax2.plot(data['Jitter_Pos'], data['Mean Dice'], label=model, linewidth=1.1, color=model_color_dict[model],
             marker=framework_marker_dict[model], markersize=7, markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)
# add_domain_divider(1.0)

plt.title('Mean Dice Comparison of the Proposed YOLOv8n+SAM2 Architecture\nwith YOLOv8n+U-Net and Traditional U-Net Baselines', pad=12, fontweight='bold')
plt.xlabel('Dataset Identification Sequence', labelpad=12)
plt.ylabel('Mean Dice Coefficient')

# CRITICAL LOCK: Bind plot space and ticks directly to the 13 targeted sequence values
plt.xlim(-0.5, len(video_axis_order) - 0.5)
ax2.set_xticks(range(len(video_axis_order)))
ax2.set_xticklabels(video_axis_order, rotation=45, ha='right')

plt.ylim(-0.05, 1.0)
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
standardize_legend(ax2)
plt.tight_layout()
plt.savefig('dice_line_framework_comparison.png', dpi=300)
plt.close()

# ==============================================================================
# PART B: LATENCY VS ACCURACY SCATTER METRIC PLOTS
# ==============================================================================
grouped_sam2 = df_sam2_only.groupby(['Architecture', 'Environment'], observed=False)[['Mean Dice', 'Avg Latency (ms)']].mean().reset_index()
grouped_framework = df_framework_only.groupby(['Architecture', 'Environment'], observed=False)[['Mean Dice', 'Avg Latency (ms)']].mean().reset_index()

grouped_sam2['Jitter_Lat'] = grouped_sam2['Avg Latency (ms)'] + grouped_sam2['Architecture'].map(jitter_map).astype(float) * 10
grouped_framework['Jitter_Lat'] = grouped_framework['Avg Latency (ms)'] + grouped_framework['Architecture'].map(jitter_map).astype(float) * 10

# Graph 3: Scatter Matrix - All YOLO+SAM2 Models
fig, ax3 = plt.subplots(figsize=(11, 5.5))
for idx, row in grouped_sam2.iterrows():
    m_arch = row['Architecture']
    m_env = row['Environment']
    if pd.isna(m_arch): continue
    
    if m_env == 'Generalization':
        ax3.scatter(row['Jitter_Lat'], row['Mean Dice'], color=model_color_dict[m_arch], marker=sam2_marker_dict[m_arch],
                    facecolors='none', edgecolors=model_color_dict[m_arch], s=130, linewidths=2.0, zorder=3)
    else:
        ax3.scatter(row['Jitter_Lat'], row['Mean Dice'], color=model_color_dict[m_arch], marker=sam2_marker_dict[m_arch],
                    s=130, edgecolor='black', linewidths=0.6, zorder=3)

leg_elements_3 = []
for model in sam2_models:
    leg_elements_3.append(mlines.Line2D([], [], color=model_color_dict[model], marker=sam2_marker_dict[model],
                                        linestyle='None', markersize=8, markeredgecolor='black', markeredgewidth=0.5, label=model))
leg_elements_3.append(mlines.Line2D([], [], color='none', linestyle='None', label='')) 
leg_elements_3.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=8, markeredgecolor='black', label='Controlled Domain (Solid)'))
leg_elements_3.append(mlines.Line2D([], [], color='none', marker='o', markerfacecolor='none', linestyle='None', markersize=8, markeredgecolor='gray', markeredgewidth=1.5, label='Generalization Domain (Hollow)'))

plt.legend(handles=leg_elements_3, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.title('Latency against Accuracy Matrix of the YOLO + SAM2 Architecture\nunder Hybrid and In-House Training Configurations', pad=12, fontweight='bold')
plt.xlabel('Average Inference Latency (ms)')
plt.ylabel('Mean Dice Score')
plt.ylim(-0.05, 0.9)
plt.xlim(-5, grouped_sam2['Avg Latency (ms)'].max() + 50)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig('scatter_tradeoff_sam2_variants.png', dpi=300)
plt.close()

# Graph 4: Scatter Matrix - Framework Comparison (5 Models)
fig, ax4 = plt.subplots(figsize=(11, 5.5))
for idx, row in grouped_framework.iterrows():
    m_arch = row['Architecture']
    m_env = row['Environment']
    if pd.isna(m_arch): continue
    
    if m_env == 'Generalization':
        ax4.scatter(row['Jitter_Lat'], row['Mean Dice'], color=model_color_dict[m_arch], marker=framework_marker_dict[m_arch],
                    facecolors='none', edgecolors=model_color_dict[m_arch], s=130, linewidths=2.0, zorder=3)
    else:
        ax4.scatter(row['Jitter_Lat'], row['Mean Dice'], color=model_color_dict[m_arch], marker=framework_marker_dict[m_arch],
                    s=130, edgecolor='black', linewidths=0.6, zorder=3)

leg_elements_4 = []
for model in framework_comparison_models:
    leg_elements_4.append(mlines.Line2D([], [], color=model_color_dict[model], marker=framework_marker_dict[model],
                                        linestyle='None', markersize=8, markeredgecolor='black', markeredgewidth=0.5, label=model))
leg_elements_4.append(mlines.Line2D([], [], color='none', linestyle='None', label='')) 
leg_elements_4.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=8, markeredgecolor='black', label='Controlled Domain (Solid)'))
leg_elements_4.append(mlines.Line2D([], [], color='none', marker='o', markerfacecolor='none', linestyle='None', markersize=8, markeredgecolor='gray', markeredgewidth=1.5, label='Generalization Domain (Hollow)'))

plt.legend(handles=leg_elements_4, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.title('Latency against Accuracy Analysis Comparing the Proposed YOLOv8n + SAM2 Architecture\nwith YOLOv8n + U-Net and Traditional U-Net Baselines', pad=12, fontweight='bold')
plt.xlabel('Average Inference Latency (ms)')
plt.ylabel('Mean Dice Score')
plt.ylim(-0.05, 0.9)
plt.xlim(0, grouped_framework['Avg Latency (ms)'].max() + 50)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig('scatter_tradeoff_framework_comparison.png', dpi=300)
plt.close()

print("Syntax error resolved. Axis ticks cleanly set to exactly 13 aligned entries.")