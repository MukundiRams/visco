import logging
logging.getLogger('numcodecs').setLevel(logging.CRITICAL)
logging.getLogger("daskms").setLevel(logging.ERROR)

import xarray as xr
import numpy as np
import dask.array as da
import dask
import os
from omegaconf import OmegaConf
from daskms import xds_to_table,xds_from_table
import visco
log = visco.get_logger(name="VISCO")




def write_subtable(zarr_path:str,msname:str,group:str):
    """
    Parameters
    ----------
    zarr_path : str
        Path to the Zarr store.
    msname : str
        Path to the output Measurement Set.
    group : str
        Group (subtable) name to write.

    Returns
    -------
    Delayed
        Dask delayed task for writing the subtable.
    """

    ds = xr.open_zarr(zarr_path,group=group)
    
    if "ROWID" in ds.coords:
        ds = ds.reset_coords("ROWID", drop=True)
        
    write = xds_to_table(ds,f"{msname}::{group}")
    
    return write


    
def list_subtables(zarr_path:str):
    """
    List all subtables (groups) in the Zarr store.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr store.

    Returns
    -------
    list
        List of subtable names.
    """
    
    return [f for f in os.listdir(zarr_path) 
            if os.path.isdir(os.path.join(zarr_path, f))]
    
def reconstruct_vis(U:np.ndarray,S:np.ndarray,Vt:np.ndarray):
    """
    Reconstruct the visibility data using the SVD components.

    Parameters
    ----------
    U : dask.array.Array
        Left singular vectors (time, mode).
    S : dask.array.Array
        Singular values (mode,).
    Vt : dask.array.Array
        Right singular vectors (mode, channel).

    Returns
    -------
    dask.array.Array
        Reconstructed visibility data (time, channel).
    """
    
    return U @ np.diag(S) @ Vt
    
def decompress_ms(zarr_path:str,msname:str,column:str):
    """
    Decompress the zarr store into an MS (MSv2).
    
    Parameters
    ------
    zarr path (str)
        The path to the zarr store.
    
    msname (str) 
        The name of the output MS.
    
    column (str)
        The column in which the compressed data is stored in in the zarr store.
    
    Returns
    -----
    None
    """
    
    main_ds = xr.open_zarr(zarr_path,group='MAIN')
    write_main = xds_to_table(main_ds,f"{msname}")
    dask.compute(write_main)
    
     
    zarr_folders  = list_subtables(zarr_path)
    tasks = []
    for folder in zarr_folders:
        if folder == 'MAIN' or 'FLAGS' or 'FLAGS_ROW':
            continue
        task = write_subtable(zarr_path,msname,folder)
        tasks.append(task)
    dask.compute(*tasks)
    
 
    antennas = xr.open_zarr(zarr_path, group='ANTENNA', consolidated=True)
    antnames = antennas.NAME.values

    maintable = xds_from_table(msname)[0]
    ant1 = maintable.ANTENNA1.values
    ant2 = maintable.ANTENNA2.values
    data_shape = maintable.DATA.shape  # (row, chan, corr)
    rowid = maintable.coords['ROWID'].values
    chunks = maintable.DATA.chunks
    

    reconstructed_data = da.zeros(data_shape, dtype=maintable.DATA.dtype,chunks=chunks)
    

    baselines = list_subtables(f"{zarr_path}/MAIN/{column}")
    

    for baseline in baselines:
        correlations = list_subtables(f"{zarr_path}/MAIN/{column}/{baseline}")
        ant1_name, ant2_name = baseline.split('&')
        try:
            ant1_idx = np.where(antnames == ant1_name)[0][0]
            ant2_idx = np.where(antnames == ant2_name)[0][0]
        except IndexError:
            log.warning(f"Baseline {baseline} not found in ANTENNA table. Skipping.")
            continue
        
        baseline_mask = (ant1 == ant1_idx) & (ant2 == ant2_idx)
        row_indices = da.where(baseline_mask)[0]
        
        for corr_idx, corr_name in enumerate(correlations):
                
            components = xr.open_zarr(f"{zarr_path}/MAIN/{column}/{baseline}/{corr_name}")
            U = components.U.data
            S = components.S.data
            Vt = components.WT.data
            
           
            vis_reconstructed = reconstruct_vis(U, S, Vt)
            reconstructed_data[row_indices, :, corr_idx] = vis_reconstructed
    
    flags_ds = xr.open_zarr(zarr_path,group='FLAGS',consolidated=True)
    flags_length = data_shape[0] * data_shape[1] * data_shape[2]
    flags = np.unpackbits(flags_ds.FLAGS.values, count=flags_length)
    flags = flags.reshape(data_shape)
    
    flag_row_ds = xr.open_zarr(zarr_path,group='FLAGS_ROW',consolidated=True)
    flags_row = np.unpackbits(flag_row_ds.FLAGS_ROW.values, count=data_shape[0])

    
    
    maintable = maintable.assign(**{
    'DATA': xr.DataArray(reconstructed_data, 
                                dims=("row", "chan", "corr"),
                                coords=
                                    {"ROWID": ("row", rowid)
                                }),
    'FLAG': xr.DataArray(flags,
                        dims=("row","chan","corr"),
                        coords={"ROWID":("row",rowid)
                                }),
    'FLAG_ROW': xr.DataArray(flags_row,
                             dims=("row"),
                             coords={"ROWID":("row",rowid)
                                     })
    })
    
    write_task = xds_to_table(maintable, f"{msname}", columns=['DATA'])
    dask.compute(write_task)        
            
    
    


    