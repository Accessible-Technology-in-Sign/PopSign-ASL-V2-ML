import numpy
import h5py

def write_file(output_file, data):
    with h5py.File(output_file, 'w') as h5_file:
        h5_file.create_dataset('intermediate', 
            data=data,
            dtype='f'
        )
