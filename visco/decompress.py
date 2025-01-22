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
    spw  = root["SPECTRAL_WINDOW"]
    
    
    
    # compressed_data_key = [key for key in root.array_keys() 
    #                      if key.startswith('COMPRESSED_DATA')][0]
    shape = root.attrs["shape"]
    chunks = root.attrs["chunks"]
    # print(f"shape:{shape}\n")
    # print(f"chunks:{chunks[0][0]}")
    # datatype = root.attrs["datatype"]
    
    decompressed_data = da.zeros(shape=shape,dtype=complex,chunks=chunks[0][0])
    
    
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
    
    num_rows = decompressed_data.shape[0]
    

    
    decompressed_xds = {
        output_column: (("row", "chan", "corr"), decompressed_data),
        "ANTENNA1": (("row",), da.from_array(root['ANTENNA1'][:],chunks=chunks[0][0])),
        "ANTENNA2": (("row",), da.from_array(root['ANTENNA2'][:],chunks=chunks[0][0])),
        "TIME": (("row",), da.from_array(root['TIME'][:],chunks=chunks[0][0])),
        "TIME_CENTROID": (("row",), da.from_array(root['TIME_CENTROID'][:],chunks=chunks[0][0])),
        "INTERVAL": (("row",), da.from_array(root['INTERVAL'][:],chunks=chunks[0][0])),
        "EXPOSURE": (("row",), da.from_array(root['EXPOSURE'][:],chunks=chunks[0][0])),
        "UVW": (("row","uvw_dim"), da.from_array(root['UVW'][:],chunks=chunks[0][0])),
        "SCAN_NUMBER": (("row",), da.from_array(root['SCAN_NUMBER'][:],chunks=chunks[0][0])),
        "FIELD_ID": (("row",), da.from_array(root['FIELD_ID'][:],chunks=chunks[0][0]))
        
    }
    
    
    
    log.info(f"Successfully decompressed visibility data from {zarr_path}")
    
    
    main_table = daskms.Dataset(
    decompressed_xds, coords={"ROWID": ("row", da.arange(num_rows,chunks=chunks[0][0]))})


    write_main = xds_to_table(main_table, ms)
    dask.compute(write_main)
    
    
    print(spw["CHAN_WIDTH"][0][:])
    spw_xds = {
        "CHAN_WIDTH":(("row"),da.from_array(spw["CHAN_WIDTH"][0][:]))
    }
    
    spw_table = daskms.Dataset(
        spw_xds
    )
    write_spw = xds_to_table(spw_table,ms)
    dask.compute(write_spw)
    
    # return decompressed_xds
        
    