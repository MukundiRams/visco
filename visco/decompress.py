from omegaconf import OmegaConf
import zarr
import numpy as np
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

#The correlation mapping.
CORR_TYPES = OmegaConf.load(f"{visco.PCKGDIR}/ms_corr_types.yaml").CORR_TYPES


def reconstruction(U,S,WT):
    """"
    Reconstruct the compressed data using the components U,S,WT.
    
    Params
    ----
    U: array
    - The left singular matrix.
    
    S: array
    - The singular values matrix.
    
    WT:array
    - The right singular matrix
    
    Returns
    ----
    The reconstructed data.
    """
    
    recon_data = U @ da.diag(S) @ WT
    
    return recon_data

def decompress_visdata(zarr_path, output_column='DATA',output_ms='decompressed.ms'):
    """
    Decompress visibility data from a Zarr file.
    
    Params
    ----
    zarr_path: str
    - The full path to the zarr file containing the compressed data.
    
    output_column:str
    - The column to put the visibility data in the MS.
    
    output_ms:str
    - The name of the MS to be created.
    
    Return
    ----
    An MS with the reconstructed data.
    
    """
    
    #Maintable and subtables of the Zarr file.
    maintable = xr.open_zarr(zarr_path)
   
    zarr_open = zarr.open(zarr_path, mode='r')
    spw = xr.open_zarr(zarr_path,group='SPECTRAL_WINDOW')
    fld = xr.open_zarr(zarr_path,group='FIELD')
    pol = xr.open_zarr(zarr_path,group='POLARIZATION')
    pointing = xr.open_zarr(zarr_path,group='POINTING')
    antenna = xr.open_zarr(zarr_path,group='ANTENNA')
    flag_row_zarr = xr.open_zarr(zarr_path,group='FLAG_ROW')
    flag_zarr = xr.open_zarr(zarr_path,group='FLAG')
    
    #Shape and chunk of the data.
    shape = maintable.attrs['shape']
    chunks = maintable.chunksizes['row'][0]
    print(f"chunks:{chunks}")
   
    #Initialize zeros.
    decompressed_data = da.zeros(shape=shape,dtype=complex,chunks=chunks)
    
    data_group_zarr = zarr_open['DATA']
    baseline_keys = list(data_group_zarr.group_keys())
    
    #Go through all the baselines and correlations.
    for baseline_key in baseline_keys:
        baseline_x = data_group_zarr[baseline_key]
        corr_keys = list(baseline_x.group_keys())
    
        for ci,corr_key in enumerate(corr_keys):
            group_path = f"DATA/{baseline_key}/{corr_key}"
            corr_group = xr.open_zarr(zarr_path, group=group_path)
            
            U = corr_group.U.data
            S = corr_group.S.data
            WT = corr_group.WT.data
            baseline_filter = corr_group.attrs['baseline_filter']
            
            reconstructed_data = reconstruction(U, S, WT)
            
            decompressed_data[baseline_filter, :, ci] = reconstructed_data
     
    nrow =  decompressed_data.shape[0] 
    nchan = decompressed_data.shape[1]
    ncorr = decompressed_data.shape[2]
    
    #Write the reconstructed data to maintable dataset.
    maintable = maintable.assign(**{
    output_column: xr.DataArray(decompressed_data, 
                                dims=("row", "chan", "corr"),
                                coords={
                                    "row": np.arange(nrow),
                                    "chan": np.arange(nchan),
                                    "corr": np.arange(ncorr)
                                })
})

    #Decompress the flag data and write it to the maintable dataset.         
    flag_row_data = flag_row_zarr.FLAG_ROW.values
    flag_row_data_unpacked = decompress_bits(flag_row_data,dim=1)  
    
    maintable = maintable.assign(**{
    "FLAG_ROW": xr.DataArray(da.from_array(flag_row_data_unpacked,chunks=chunks), 
                                dims=("row"),
                                coords={
                                    "row": np.arange(nrow),
                                }) 
    })      
    
    
    flag_data = flag_zarr.FLAG.values
    flag_data_unpacked = decompress_bits(flag_data,dim=3,nrow=nrow,nchan=nchan,ncorr=ncorr)  
    
    maintable = maintable.assign(**{
    "FLAG": xr.DataArray(da.from_array(flag_data_unpacked,chunks=chunks), 
                                dims=("row", "chan", "corr"),
                                coords={
                                    "row": np.arange(nrow),
                                    "chan": np.arange(nchan),
                                    "corr": np.arange(ncorr)
                                }) 
    })      
    
    
    #Finally, write the MS.
    write_main = xds_to_table(maintable, output_ms)
    with TqdmCallback(desc=f'Writing Main Table to {output_ms}'):
        
        dask.compute(write_main)   
        
    write_spw = xds_to_table(spw, f"{output_ms}::SPECTRAL_WINDOW")
    with TqdmCallback(desc=f'Writing SPECTRAL_WINDOW table to {output_ms}'):
        
        dask.compute(write_spw)
        
        
    write_fld = xds_to_table(fld, f"{output_ms}::FIELD")
    with TqdmCallback(desc=f'Writing FIELD table to {output_ms}'):
        
        dask.compute(write_fld)
        
    write_pol = xds_to_table(pol, f"{output_ms}::POLARIZATION")
    with TqdmCallback(desc=f'Writing POLARIZATION table to {output_ms}'):
        
        dask.compute(write_pol)
    
    write_antenna = xds_to_table(antenna, f"{output_ms}::ANTENNA")
    with TqdmCallback(desc=f'Writing ANTENNA table to {output_ms}'):
        
        dask.compute(write_antenna)
        
    write_pointing = xds_to_table(pointing, f"{output_ms}::POINTING")
    with TqdmCallback(desc=f'Writing POINTING table to {output_ms}'):
        
        dask.compute(write_pointing)
        
  
    
def decompress_bits(data_array,dim=1, nrow=None,nchan=None,ncorr=None):
    """"
    Decompress data (flag) using numpy unpackbits.
    
    Params
    ----
    data_array: array
    - The data array to be compressed.
    
    dim: int
    - The number of dimensions of the data.
    
    nrow: int
    - The number of rows of the visibility data.
    
    nchan: int
    - The number of channels of the visibility data.
    
    ncorr: int
    - THe number of correlations of the visibility data.
    """
    
    #If the data has one dimension, its easy. Just make sure we account for padding by 
    #selecting the first nrow.
    if dim==1:
        unpacked_data = np.unpackbits(data_array,axis=None)
        unpacked_data = unpacked_data[:nrow]
        
    #If the data has more than one dimension, we have to reshape the data to the original because
    #it has been flattened.
    elif dim>1:
        undone_data = np.unpackbits(data_array,axis=None)
        undone_data = undone_data[:nrow*nchan*ncorr]
        unpacked_data = undone_data.reshape(nrow,nchan,ncorr)
    return unpacked_data    

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