import pandas as pd
import os

def filter_shortlisted_data(full_data_path, shortlist_txt_path):
    try:
        # Load the full CSV data
        # header=0 tells pandas that the first row contains column names
        df_full = pd.read_csv(full_data_path, sep=',', engine='python', header=0)
        
        # We need to know the name of the first column to filter by it
        # This is usually the column containing the .jpg names
        id_column_name = df_full.columns[0]
        
        # Clean up any accidental whitespace in that first column
        df_full[id_column_name] = df_full[id_column_name].astype(str).str.strip()
        
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return

    # Load the shortlist names
    with open(shortlist_txt_path, 'r', encoding='utf-8') as f:
        shortlist = [line.strip() for line in f if line.strip()]

    # Filter: Keep rows where the ID column matches our shortlist
    filtered_df = df_full[df_full[id_column_name].isin(shortlist)]

    # --- Naming Logic ---
    output_dir = os.path.dirname(shortlist_txt_path)
    shortlist_base_name = os.path.splitext(os.path.basename(shortlist_txt_path))[0]
    
    output_name = f"{shortlist_base_name}_shortlistedData.csv"
    output_path = os.path.join(output_dir, output_name)

    # Save the result
    # header=True ensures the column names (z, pose, force, etc.) are saved in Row 1
    filtered_df.to_csv(output_path, sep=',', index=False, header=True)
    
    print(f"--- Process Complete ---")
    print(f"Header Detected: {list(df_full.columns)}")
    print(f"Saved to: {output_name}")
    print(f"Total Rows: {len(filtered_df) + 1} (1 header + {len(filtered_df)} images)")

# --- YOUR PATHS ---
full_csv = r'/home/athena/Ultrasound_videos/human_test_results/Process_Data/Data_P_Human_Throat_US/cgl/filtered/20250930122939/delta_pose_force.csv'
shortlist_txt = r'/home/athena/Ultrasound_videos/human_test_results/Process_Data/Data_P_Human_Throat_US/cgl/filtered/20250930122939/20250930122939.txt'

filter_shortlisted_data(full_csv, shortlist_txt)