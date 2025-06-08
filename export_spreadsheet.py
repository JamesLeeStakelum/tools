"""
Excel to Text File Processor

This script connects to an Excel spreadsheet (.xlsx) and extracts data from
specific columns (A, H, M, R) from each row, starting from the third row.
Each row's extracted data is then written to individual text files
(question.txt, answer1.txt, answer2.txt, answer3.txt) within a structured
folder system (sets/setX/). Processing stops when an empty cell is
encountered in column A. Empty cells in other columns will be replaced
with the placeholder "[EMPTY]".

Dependencies:
    - openpyxl: Required for reading .xlsx Excel files.

Installation:
    To install the necessary library, run the following pip command:
    pip install openpyxl

Usage:
    Run the script from your terminal:
    python your_script_name.py <path_to_your_excel_file.xlsx>

    Example:
    python excel_processor.py my_data.xlsx

    You can also specify a custom output directory:
    python excel_processor.py my_data.xlsx --output-dir my_extracted_sets
"""

import openpyxl
import os
import argparse # Import the argparse module for command-line arguments

def process_excel_to_text_files(excel_file_path, output_base_dir="sets"):
    """
    Connects to an Excel spreadsheet (.xlsx), skips the first two header rows,
    extracts values from columns A, H, M, and R for each subsequent row,
    and writes them to respective text files within a structured folder system.
    Processing stops when a row has no value in column A.

    Args:
        excel_file_path (str): The path to the Excel (.xlsx) file.
        output_base_dir (str): The name of the base directory where 'setX' folders will be created.
                               Defaults to 'sets'.
    """
    # Create the main output directory if it doesn't exist
    # Using os.getcwd() to ensure the base directory is relative to the script's execution location.
    base_output_dir_full_path = os.path.join(os.getcwd(), output_base_dir)
    os.makedirs(base_output_dir_full_path, exist_ok=True)

    # Define base output file names (will be placed in set-specific subfolders)
    question_filename = "question.txt"
    answer1_filename = "answer1.txt"
    answer2_filename = "answer2.txt"
    answer3_filename = "answer3.txt"

    # Load the workbook and select the active sheet
    # Using read_only=True for performance and data_only=True to get computed values from formulas.
    workbook = openpyxl.load_workbook(filename=excel_file_path, read_only=True, data_only=True)
    sheet = workbook.active

    # Iterate through rows, starting from the third row (min_row=3)
    # row_index starts from 3 because min_row=3, representing the actual Excel row number.
    # values_only=True is maintained for direct value access in the loop.
    for row_index, row_data in enumerate(sheet.iter_rows(min_row=3, values_only=True), start=3):
        # Columns are 0-indexed when accessed via row_data tuple from values_only=True
        # 'A' is index 0
        # 'H' is index 7
        # 'M' is index 12
        # 'R' is index 17

        # Extract value from Column A first to check for stopping condition
        # Check if the row has at least column A (index 0) to avoid IndexError on empty rows
        question = row_data[0] if len(row_data) > 0 else None

        # Stopping Condition: Stop if Column A is empty (None or empty string after stripping)
        # This check is robust for various empty cell representations.
        if question is None or (isinstance(question, str) and question.strip() == ""):
            # End of data reached, stop processing
            break # Exit the loop

        # Calculate the set number and create the set-specific directory
        # Since Excel row 3 is set1, Excel row 4 is set2, etc.
        set_number = row_index - 2
        current_set_dir = os.path.join(base_output_dir_full_path, f"set{set_number}")
        os.makedirs(current_set_dir, exist_ok=True)

        # Helper function to get cell value or [EMPTY] placeholder
        def get_value_or_empty_placeholder(value):
            return str(value) if value is not None and str(value).strip() != "" else "[EMPTY]"

        # Extract values from the specified columns for the current row/set
        # Using a safe way to access, converting None or empty string to [EMPTY].
        # For question_val, we use the raw 'question' variable as it's already handled by the break condition.
        question_val = str(question) if question is not None else "[EMPTY]"
        answer1_val = get_value_or_empty_placeholder(row_data[7]) if len(row_data) > 7 else "[EMPTY]" # Column 'H'
        answer2_val = get_value_or_empty_placeholder(row_data[12]) if len(row_data) > 12 else "[EMPTY]" # Column 'M'
        answer3_val = get_value_or_empty_placeholder(row_data[17]) if len(row_data) > 17 else "[EMPTY]" # Column 'R'

        # Construct full paths for the output files for the current set
        q_file_path = os.path.join(current_set_dir, question_filename)
        a1_file_path = os.path.join(current_set_dir, answer1_filename)
        a2_file_path = os.path.join(current_set_dir, answer2_filename)
        a3_file_path = os.path.join(current_set_dir, answer3_filename)

        # Open all text files in WRITE mode ('w') for the current set
        # This ensures that each file for a given set is overwritten if it exists.
        with open(q_file_path, 'w', encoding='utf-8') as q_f, \
             open(a1_file_path, 'w', encoding='utf-8') as a1_f, \
             open(a2_file_path, 'w', encoding='utf-8') as a2_f, \
             open(a3_file_path, 'w', encoding='utf-8') as a3_f:

            q_f.write(question_val)
            a1_f.write(answer1_val)
            a2_f.write(answer2_val)
            a3_f.write(answer3_val)


# --- Main execution block with argparse ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extracts questions and answers from an XLSX file into organized text file sets.'
    )
    parser.add_argument(
        'xlsx_path',
        help='Path to the .xlsx file to be processed.'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='sets',
        help='Optional: Base directory name for output sets (default: "sets").'
    )
    args = parser.parse_args()

    # Call the processing function with arguments from the command line
    process_excel_to_text_files(args.xlsx_path, args.output_dir)
