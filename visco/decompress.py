from omegaconf import OmegaConf
import zarr
import dask, daskms
import numpy as np
from daskms import xds_to_table
import xarray as xr
import visco
import dask.array as da
from casacore.tables import table
log = visco.get_logger(name="VISCO")
import logging
from tqdm.dask import TqdmCallback
logging.getLogger("daskms").setLevel(logging.ERROR)






CORR_TYPES = OmegaConf.load(f"{visco.PCKGDIR}/ms_corr_types.yaml").CORR_TYPES

def reconstruction(U,S,WT):
    """"
    Reconstruct the compressed data.
    """
    
    recon_data = U @ da.diag(S) @ WT
    
    return recon_data





def decompress_visdata(zarr_path, output_column='DATA',output_ms='decompressed.ms'):
    """
    Decompress visibility data from a Zarr file.
    
    """
 
    root = zarr.open(zarr_path, mode='r')
    spw  = root["SPECTRAL_WINDOW"]
    pol  = root["POLARIZATION"]
    ant  = root["ANTENNA"]
    field = root["FIELD"]
    pointing = root["POINTING"]

    # compressed_data_key = [key for key in root.array_keys() 
    #                      if key.startswith('COMPRESSED_DATA')][0]
    shape = root.attrs["shape"]
    chunks = root.attrs["chunks"]

    
    decompressed_data = da.zeros(shape=shape,dtype=complex,chunks=chunks[0][0])
    
    decomp_group = root['DECOMPOSITIONS']
    
    first_baseline = list(decomp_group.group_keys())[0]
    
    corr_list = list(decomp_group[first_baseline].group_keys())
    
    
    corr_indices = []
    for corr in corr_list:
        corr_indices.append(CORR_TYPES[corr])
    
    corr_to_index = {corr: idx for idx, corr in enumerate(corr_list)}
    
    for baseline_key in decomp_group.group_keys():
        baseline_group = decomp_group[baseline_key]
        
        for corr in baseline_group.group_keys():
            
            log.info(f"Processing baseline {baseline_key}, correlation {corr}")
            corr_group = baseline_group[corr]
            
            
            U = corr_group['U'][:]
            singvals = corr_group['S'][:]
            WT = corr_group['WT'][:]

            corr_index = corr_to_index[corr]

            baseline_filter = np.array(corr_group.attrs['baseline_filter'])
    
            decompressed_visdata = reconstruction(U, singvals, WT)
            
            
            decompressed_data[baseline_filter, :, corr_index] = decompressed_visdata
            
    
    num_rows = decompressed_data.shape[0]
    
    #Main Table
    
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
    
    
    # log.info(f"Writing Main Table to {output_ms}")
    
    main_table = daskms.Dataset(
    decompressed_xds, coords={"ROWID": ("row", da.arange(num_rows,chunks=chunks[0][0]))})
    write_main = xds_to_table(main_table, output_ms)
    with TqdmCallback(desc=f'Writing Main Table to {output_ms}'):
        
        dask.compute(write_main)
    
    
    
    # log.info(f"Writing SPECTRAL WINDOW table to {output_ms}")
    
    chan_width = spw["CHAN_WIDTH"][0][:].reshape(1,spw["CHAN_WIDTH"][0][:].shape[0])
    chan_freq = spw["CHAN_FREQ"][0][:].reshape(1,spw["CHAN_FREQ"][0][:].shape[0])
    effective_bw = spw["EFFECTIVE_BW"][0][:].reshape(1,spw["EFFECTIVE_BW"][0][:].shape[0])
    resolution = spw["RESOLUTION"][0][:].reshape(1,spw["RESOLUTION"][0][:].shape[0])
    num_chan = [spw["NUM_CHAN"][0]]
    ref_freq = [spw["REF_FREQUENCY"][0]]
    meas_freq_ref = [spw["MEAS_FREQ_REF"][0]]
    total_bandwidth = [spw["TOTAL_BANDWIDTH"][0]]
    flag_row = [spw["FLAG_ROW"][0]]
    spw_xds = {
        "CHAN_FREQ":(("row","chan"),da.from_array(chan_freq)),
        "CHAN_WIDTH":(("row","chan"),da.from_array(chan_width)),
        "EFFECTIVE_BW":(("row","chan"),da.from_array(effective_bw)),
        "RESOLUTION":(("row","chan"),da.from_array(resolution)),
        "NUM_CHAN":(("row",),da.from_array(num_chan)),
        "REF_FREQUENCY":(("row",),da.from_array(ref_freq)),
        "MEAS_FREQ_REF":(("row",),da.from_array(meas_freq_ref)),
        "TOTAL_BANDWIDTH":(("row",),da.from_array(total_bandwidth)),
        "FLAG_ROW":(("row",),da.from_array(flag_row))
        
    }
    
    spw_table = daskms.Dataset(
        spw_xds)
    write_spw = xds_to_table(spw_table,f"{output_ms}::SPECTRAL_WINDOW")
    with TqdmCallback(desc=f"Writing SPECTRAL WINDOW table to {output_ms}"):
        dask.compute(write_spw)
        
     #polarization table
    pol_xds = {
         "CORR_TYPE":(("row","corr"), da.from_array(pol["CORR_TYPE"][:])),
         "CORR_PRODUCT":(("row","corr"), da.from_array(pol["CORR_PRODUCT"][:])),
         "NUM_CORR":(("row",), da.from_array(pol["NUM_CORR"]))
        }   
     
    pol_table = daskms.Dataset(
        pol_xds)
    write_pol = xds_to_table(pol_table,f"{output_ms}::POLARIZATION")
    with TqdmCallback(desc=f"Writing POLARIZATION table to {output_ms}"):
        dask.compute(write_pol)
    
    
    
   
    log.info(f"Successfully decompressed visibility data from {zarr_path}")  
    

def ms_addrow(ms,subtable,nrows):
    
    subtab = table(f"{ms}::{subtable}",
                    readonly=False, lockoptions='user', ack=False)
    try:
        subtab.lock(write=True)
        subtab.addrows(nrows)
        
    finally:
        subtab.unlock()
        subtab.close()
        
       
def ms_remrow(ms,subtable,row):
    
    subtab = table(f"{ms}::{subtable}",
                    readonly=False, lockoptions='user', ack=False)
    try:
        subtab.lock(write=True)
        subtab.removerows(row)
        
    finally:
        subtab.unlock()
        subtab.close()