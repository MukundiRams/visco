import zarr
import dask, daskms
import numpy as np
import xarray as xr
import visco
from visco.compress import reconstruction
log = visco.get_logger(name="BDSVD")

def decompress_visdata(zarr_path, output_column='DECOMPRESSED_DATA'):
    """
    Decompress visibility data from a hierarchically organized Zarr store.
    
    Parameters
    ----------
    zarr_path : str
        Path to the Zarr store containing compressed data and decomposition components
    output_column : str, optional
        Name of the output column for decompressed data
        
    Returns
    -------
    xarray.Dataset
        Dataset containing the decompressed visibility data
    """
    
    try:
       
        print(zarr_path)
        root = zarr.open(zarr_path, mode='r')
        
        
        compression_rank = root.attrs['compression_rank']
        
        
        compressed_data_key = [key for key in root.array_keys() 
                             if key.startswith('COMPRESSED_DATA')][0]
        decompressed_data = np.zeros_like(root[compressed_data_key])
        
        
        decomp_group = root['DECOMPOSITIONS']
        
        
        for baseline_key in decomp_group.group_keys():
            baseline_group = decomp_group[baseline_key]
            
            
            for corr in baseline_group.group_keys():
                corr_group = baseline_group[corr]
                
                
                U = corr_group['U'][:]
                singvals = corr_group['singvals'][:]
                WT = corr_group['WT'][:]
                
                
                baseline_filter = np.array(corr_group.attrs['baseline_filter'])
                ci = corr_group.attrs['ci']
                fullrank = corr_group.attrs['fullrank']
                
                
                decompressed_visdata = reconstruction(U, singvals, WT, fullrank, compression_rank, 'cum')
                
                
                decompressed_data[baseline_filter, :, ci] = decompressed_visdata
                
                log.info(f"Decompressed visibility data for baseline {baseline_key}, correlation {corr}")
        
        
        decompressed_xds = xr.Dataset({
            output_column: (("row", "chan", "corr"), decompressed_data),
            "ANTENNA1": (("row",), root['ANTENNA1'][:]),
            "ANTENNA2": (("row",), root['ANTENNA2'][:])
        })
        
        log.info(f"Successfully decompressed visibility data from {zarr_path}")
        return decompressed_xds
        
    except Exception as e:
        log.error(f"Failed to decompress visibility data from {zarr_path}: {e}")
        raise