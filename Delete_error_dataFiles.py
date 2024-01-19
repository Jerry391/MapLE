import os
import pandas as pd

def get_first_four_letters(filename):
    return filename[:4]

if __name__ == "__main__":
    current_path = os.getcwd()  # Get the current working directory
    target_folder = os.path.join(current_path, "test3", "data")

    excel_file_path = os.path.join(current_path, "test3", "Optimal_set2.xlsx")
    excel_data = pd.read_excel(excel_file_path)
    first_column_values = excel_data.iloc[:, 0].tolist()

    files_to_keep = []

    for root, dirs, files in os.walk(target_folder):
        for filename in files:
            if filename.endswith(".mol2"):  # Only consider Excel files
                file_path = os.path.join(root, filename)
                if get_first_four_letters(filename) in first_column_values:
                    files_to_keep.append(file_path)
                else:
                    os.remove(file_path)  # Delete files that don't match

    print(f"Files matching the criteria have been retained, and others have been deleted.")
