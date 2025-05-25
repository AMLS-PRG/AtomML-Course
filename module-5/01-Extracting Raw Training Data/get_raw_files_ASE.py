import os
import numpy as np
import ase.io
import dpdata

# Input and output settings
input_dir = "./QE"  # Modify to your actual path
base_output_dir = "./"                   # Directory to save DeepMD raw files
threshold_size = 90000  # File size threshold (in bytes) to filter converged outputs


def read_qe_out_single(file_path, types_mapping=None):
    """
    Read a single Quantum ESPRESSO .out file and extract positions, energy, forces, cell, and atom types.

    Args:
        file_path (str): Path to the .out file
        types_mapping (dict): Mapping from element symbols to type indices, e.g., {"Ca": "0", "C": "1", "O": "2"}

    Returns:
        dict: A dictionary with keys 'positions', 'energy', 'forces', 'cell', 'types'
    """
    try:
        conf = ase.io.read(file_path, format='espresso-out')
    except Exception as e:
        print(f"Cannot read file {file_path}: {e}")
        return None

    try:
        forces = conf.get_forces()
    except Exception as e:
        print(f"Forces missing in file {file_path}: {e}")
        return None

    data = {
        'positions': conf.get_positions(),
        'energy': conf.get_potential_energy(),
        'forces': forces,
        'cell': conf.get_cell(),
    }

    if types_mapping is not None:
        symbols = np.array(conf.get_chemical_symbols())
        mapped_types = np.array([types_mapping[symbol] for symbol in symbols])
        data['types'] = mapped_types
    else:
        data['types'] = np.array(conf.get_chemical_symbols())

    return data


######################################## Main Script ########################################

# Create a new output folder: itaddXX
index = 1
while True:
    output_dir = os.path.join(base_output_dir, f"itadd{index:02d}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        break
    index += 1

# Element-to-type mapping
types_mapping = {"K": "0", "Al": "1", "Si": "2", "O": "3", "H": "4"}

# Collect valid data
all_data = []

for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)

    if filename.startswith("pw-") and filename.endswith(".out") and os.path.getsize(file_path) > threshold_size:
        data = read_qe_out_single(file_path, types_mapping=types_mapping)

        if data is not None:
            all_data.append(data)
            print(f"Converted and added: {filename}")
        else:
            print(f"Skipped {filename} due to error.")

# Write raw files
with open(os.path.join(output_dir, "coord.raw"), "w") as file_coord, \
     open(os.path.join(output_dir, "energy.raw"), "w") as file_energy, \
     open(os.path.join(output_dir, "force.raw"), "w") as file_force, \
     open(os.path.join(output_dir, "box.raw"), "w") as file_box, \
     open(os.path.join(output_dir, "type.raw"), "w") as file_type:

    for data in all_data:
        file_coord.write(' '.join(data['positions'].flatten().astype(str).tolist()) + '\n')
        file_energy.write(str(data['energy']) + '\n')
        file_force.write(' '.join(data['forces'].flatten().astype(str).tolist()) + '\n')
        file_box.write(' '.join(data['cell'].flatten().astype(str).tolist()) + '\n')

    # Write atom types (assumed the same for all frames)
    file_type.write(' '.join(all_data[0]['types'].tolist()) + '\n')

# Write type_map.raw (element order based on type index)
elements_sorted = sorted(types_mapping.keys(), key=lambda x: int(types_mapping[x]))
with open(os.path.join(output_dir, "type_map.raw"), "w") as f_map:
    for elem in elements_sorted:
        f_map.write(elem + "\n")

# Convert to DeepMD npy format
system = dpdata.LabeledSystem(output_dir, fmt='deepmd/raw')
system.to_deepmd_npy(output_dir)

print(f"\nConversion completed. Raw and NPY files saved to: {output_dir}")
