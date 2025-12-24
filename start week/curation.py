import pandas as pd
import os
from pathlib import Path


def combine_tsv_files():
    # 1. Calculate the path to the 'datasets' folder
    # We are in src/dreambank/curation.py, so we go up 2 levels to find the project root
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parents[2]
    datasets_folder = project_root / "datasets"

    print(f"Looking for TSV files in: {datasets_folder}")

    if not datasets_folder.exists():
        print("Error: The 'datasets' folder was not found. Check the path.")
        return

    all_dataframes = []

    # 2. Loop through all files in the folder
    # We use glob("*.tsv") to grab only the TSV files, ignoring JSONs
    tsv_files = list(datasets_folder.glob("*.tsv"))

    if not tsv_files:
        print("No .tsv files found in the folder.")
        return

    print(f"Found {len(tsv_files)} TSV files. Processing...")

    for file_path in tsv_files:
        try:
            # 3. Read the TSV file (tab-separated)
            # on_bad_lines='skip' helps if some files have formatting errors
            df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')

            # Optional: Add a column to know which file the dream came from
            df['source_filename'] = file_path.name

            all_dataframes.append(df)
            print(f" - Loaded {file_path.name} ({len(df)} dreams)")
        except Exception as e:
            print(f" ! Failed to read {file_path.name}: {e}")

    # 4. Combine everything into one
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Save to the project root
        output_file = project_root / "all_dreams_combined.csv"
        combined_df.to_csv(output_file, index=False)

        print("-" * 30)
        print(f"SUCCESS! Combined {len(combined_df)} dreams.")
        print(f"File saved at: {output_file}")
    else:
        print("No data could be combined.")


if __name__ == "__main__":
    combine_tsv_files()