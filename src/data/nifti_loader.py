import nibabel as nib
import dask.array as da

def load_nifti(filepath, chunk_size=(100, 100, 100)):
    """
    Load a NIfTI file and return a Dask array with the specified chunk size.
    """
    img = nib.load(filepath)
    data = img.get_fdata()
    # Create a Dask array with given chunk size for efficient out-of-core processing.
    dask_data = da.from_array(data, chunks=chunk_size)
    return dask_data
