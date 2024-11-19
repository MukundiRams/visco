import zarr
import dask, daskms
import numpy as np
from daskms import xds_to_table
import xarray as xr
import visco
import dask.array as da
log = visco.get_logger(name="BDSVD")


def reconstruction(U,S,WT):
    """"
    Reconstruct the compressed data.
    """
    
    recon_data = U @ da.diag(S) @ WT
    
    return recon_data

def decompress_visdata(zarr_path, output_column='DECOMPRESSED_DATA',ms='decompressed.ms'):
    """
    Decompress visibility data from a Zarr file.
    
    """
    
    
       
    root = zarr.open(zarr_path, mode='r')
    
    
    
    # compressed_data_key = [key for key in root.array_keys() 
    #                      if key.startswith('COMPRESSED_DATA')][0]
    shape = root.attrs["shape"]
    # datatype = root.attrs["datatype"]
    
    decompressed_data = da.zeros(shape=shape,dtype=complex)
    
    
    decomp_group = root['DECOMPOSITIONS']
    
    
    for baseline_key in decomp_group.group_keys():
        baseline_group = decomp_group[baseline_key]
        
        
        for corr in baseline_group.group_keys():
            corr_group = baseline_group[corr]
            
            
            U = corr_group['U'][:]
            singvals = corr_group['S'][:]
            WT = corr_group['WT'][:]
            
            
            baseline_filter = np.array(corr_group.attrs['baseline_filter'])
            ci = corr_group.attrs['ci']
            
            
            decompressed_visdata = reconstruction(U, singvals, WT)
            
            
            decompressed_data[baseline_filter, :, ci] = decompressed_visdata
            
            log.info(f"Decompressed visibility data for baseline {baseline_key}, correlation {corr}")
    
    
    decompressed_xds = {
        output_column: (("row", "chan", "corr"), decompressed_data),
        "ANTENNA1": (("row",), da.from_array(root['ANTENNA1'][:])),
        "ANTENNA2": (("row",), da.from_array(root['ANTENNA2'][:])),
        "TIME": (("row",), da.from_array(root['TIME'][:])),
        "TIME_CENTROID": (("row",), da.from_array(root['TIME_CENTROID'][:])),
        "INTERVAL": (("row",), da.from_array(root['INTERVAL'][:])),
        "EXPOSURE": (("row",), da.from_array(root['EXPOSURE'][:])),
        "UVW": (("row","uvw_dim"), da.from_array(root['UVW'][:])),
        "SCAN_NUMBER": (("row",), da.from_array(root['SCAN_NUMBER'][:])),
        "FIELD_ID": (("row",), da.from_array(root['FIELD_ID'][:]))
        
    }
    
    
    
    log.info(f"Successfully decompressed visibility data from {zarr_path}")
    
    main_table = daskms.Dataset(
    decompressed_xds, coords={"ROWID": ("row", da.arange(decompressed_data.shape[0]))})


    write_main = xds_to_table(main_table, ms)
    dask.compute(write_main)
    
    # return decompressed_xds
        
    