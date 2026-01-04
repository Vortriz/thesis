import os
import shutil
import subprocess
import sys
from pathlib import Path


def recursive_copy_and_process(source_dir):
    """
    Recursively copies a source directory to a destination directory,
    then runs a specified terminal command on all .ipynb files
    found in the *copied* directory.
    """
    if not Path.is_dir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    dest_dir = source_dir + "-marimo"

    # --- Step 1: Copy the folder recursively ---
    print(f"Attempting to copy '{source_dir}' to '{dest_dir}'...")

    # `dirs_exist_ok=False` ensures we don't accidentally overwrite an existing directory.
    # If the destination should be overwritten, `dirs_exist_ok=True` could be used,
    # but the default behavior is safer for a first run.
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=False)
    print(f"Successfully copied '{source_dir}' to '{dest_dir}'.")

    # --- Step 2: Traverse the copied folder and run the command on .ipynb files ---
    print(f"\nProcessing .ipynb files in '{dest_dir}'...")
    ipynb_files_processed_count = 0
    errors_during_processing = []

    for root, _, files in os.walk(dest_dir):
        for file in files:
            if file.lower().endswith(".ipynb"):
                ipynb_files_processed_count += 1
                file_name = file.split(".")[0]  # Get the file name without extension
                # Construct the full command, ensuring the file path is quoted
                # to handle spaces or special characters in paths.
                source = str(Path(root) / f"{file_name}.ipynb")
                destination = str(Path(root) / f"{file_name}.py")
                full_command = f'uv run marimo convert "{source}" -o "{destination}"'

                print(f"  Running command on: {file}")

                result = subprocess.run(
                    full_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",  # Explicitly set encoding for consistent output
                    errors="replace",  # Replace unencodable characters
                )
                if result.stdout.strip():
                    print(f"    STDOUT:\n{result.stdout.strip()}")
                if result.stderr.strip():
                    print(f"    STDERR:\n{result.stderr.strip()}")
                print(f"    Successfully processed '{file}'.")

    print("\n--- Processing Summary ---")
    if ipynb_files_processed_count == 0:
        print(f"No .ipynb files were found in the copied directory '{dest_dir}'.")
    elif errors_during_processing:
        print(
            f"Finished with {len(errors_during_processing)} errors out of {ipynb_files_processed_count} .ipynb files processed.",
        )
        print("Files that encountered errors:")
        for err_file in errors_during_processing:
            print(f"  - {err_file}")
    else:
        print(f"Successfully processed all {ipynb_files_processed_count} .ipynb files.")


if __name__ == "__main__":
    source_folder = sys.argv[1]

    recursive_copy_and_process(source_folder)
