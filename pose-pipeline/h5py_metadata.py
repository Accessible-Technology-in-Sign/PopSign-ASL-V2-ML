import numpy
import h5py


# Writes additional metadata about the file as well
def write_file(output_file, data, metadata):
    with h5py.File(output_file, 'w') as h5_file:
        # Create the dataset
        dataset = h5_file.create_dataset('intermediate', 
            data=data,
            dtype='f'
        )
        
        # Add metadata as attributes
        for key, value in metadata.items():
            dataset.attrs[key] = value