import os
import dpdata

# Set input and output directories
input_dir = "./QE"  # Modify this to your actual input path
base_output_dir = "./"                     # Directory to save DeepMD raw files
threshold_size = 90000  # File size threshold (in bytes) to filter converged calculations

# Automatically create a new directory: itaddXX
index = 1
while True:
    output_dir = os.path.join(base_output_dir, f"itadd{index:02d}")
    if not os.path.exists(output_dir):  # Find the first available directory name
        os.makedirs(output_dir)
        break
    index += 1

# Initialize an empty LabeledSystem dataset
merged_system = dpdata.LabeledSystem()

# Counter to keep track of successfully converted files
converted_count = 0

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)

    # Process only pw-*.out files that are larger than the threshold (indicating convergence)
    if filename.startswith("pw-") and filename.endswith(".out") and os.path.getsize(file_path) > threshold_size:
        try:
            # Load QE calculation output
            system = dpdata.LabeledSystem(file_path, fmt='qe/pw/scf')

            # Append to the merged dataset
            merged_system.append(system)

            converted_count += 1
            print(f"Converted and added: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save the merged dataset if any files were successfully converted
if converted_count > 0:
    merged_system.to_deepmd_npy(output_dir)
    merged_system.to_deepmd_raw(output_dir)
    print(f"\nMerged dataset saved to {output_dir}")
else:
    print("\nNo valid QE output files were converted.")

# Print final conversion summary
print(f"\nConversion completed! Successfully merged {converted_count} files.")
