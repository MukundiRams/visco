import logging
logging.getLogger('numcodecs').setLevel(logging.CRITICAL)
logging.getLogger("daskms").setLevel(logging.ERROR)

import xarray as xr
import numpy as np
import dask.array as da
import dask
from dask import delayed
import os
import shutil
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


def unstack_vis(vis_reconstructed, nrows):
    """Return list of blocks each with shape (nrows, nchan)."""
    if isinstance(vis_reconstructed, da.Array):
        nstack = int(vis_reconstructed.shape[0] // nrows)
        return [vis_reconstructed[i*nrows:(i+1)*nrows, :] for i in range(nstack)]
    else:
        nstack = vis_reconstructed.shape[0] // nrows
        return list(np.split(vis_reconstructed, nstack, axis=0))



def construct_main_ds(zarr_path:str,column:str):
    """
    Construct the full main table dataset.
    
    Parameters
    ------
    zarr path (str)
        The path to the zarr store.
    
    column (str)
        The column in which the compressed data is stored in in the zarr store.
    
    Returns
    -----
    maintable dataset (xarray dataset)
    """
    
    maintable = xr.open_zarr(zarr_path,group='MAIN',consolidated=True)
    antennas = xr.open_zarr(zarr_path, group='ANTENNA', consolidated=True)
    antnames = antennas.NAME.values

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
        row_indices = da.where(baseline_mask)[0].compute()
        # nrows = row_indices.compute()
        nrows=row_indices.size
        corr_indices = {'XX':0,'XY':1, 'YX':2,'YY':-1}
        
        for corr_idx, corr_name in enumerate(correlations):
            
            components = xr.open_zarr(f"{zarr_path}/MAIN/{column}/{baseline}/{corr_name}")
            U = components.U.data
            S = components.S.data
            Vt = components.WT.data
            
            vis_reconstructed = reconstruct_vis(U, S, Vt)
            
            if corr_name == 'diagonals':
                parts = unstack_vis(vis_reconstructed, nrows)
                reconstructed_data[row_indices, :, 0] = parts[0]
                reconstructed_data[row_indices, :, 3] = parts[1]
                
                
            elif corr_name == 'offdiagonals':
                parts = unstack_vis(vis_reconstructed, nrows)
                reconstructed_data[row_indices, :, 1] = parts[0]
                reconstructed_data[row_indices, :, 2] = parts[1]
            
            else:
                reconstructed_data[row_indices, :, corr_indices[corr_name]] = vis_reconstructed
    
    flags_ds = xr.open_zarr(zarr_path,group='FLAGS',consolidated=True)
    flags_length = data_shape[0] * data_shape[1] * data_shape[2]
    flags = np.unpackbits(flags_ds.FLAGS.values, count=flags_length)
    flags = flags.reshape(data_shape)
    
    flag_row_ds = xr.open_zarr(zarr_path,group='FLAGS_ROW',consolidated=True)
    flags_row = np.unpackbits(flag_row_ds.FLAGS_ROW.values, count=data_shape[0])

    if 'WEIGHT_SPECTRUM' in list_subtables(f"{zarr_path}"):
        weights = xr.open_zarr(f"{zarr_path}/WEIGHT_SPECTRUM",consolidated=True)

        
        weights_reconstructed = reconstruct_vis(weights.U.values,weights.S.values,weights.WT.values)
        weights_expanded = np.expand_dims(weights_reconstructed,axis=-1)
        final_weights = np.tile(weights_expanded,(1,1,data_shape[2]))
        
        maintable = maintable.assign(**{
            'WEIGHT_SPECTRUM':xr.DataArray(da.from_array(final_weights,chunks=chunks),
                                        dims=("row","chan","corr"),
                                        coords={"ROWID":("row",rowid)
                                                }),
            'SIGMA_SPECTRUM':xr.DataArray(da.from_array(final_weights,chunks=chunks),
                                        dims=("row","chan","corr"),
                                        coords={"ROWID":("row",rowid)
                                                })
                })
        
    
    
    maintable = maintable.assign(**{
    'DATA': xr.DataArray(reconstructed_data, 
                                dims=("row", "chan", "corr"),
                                coords=
                                    {"ROWID": ("row", rowid)
                                }),
    'FLAG': xr.DataArray(da.from_array(flags,chunks=chunks),
                        dims=("row","chan","corr"),
                        coords={"ROWID":("row",rowid)
                                }),
    'FLAG_ROW': xr.DataArray(da.from_array(flags_row,chunks=chunks[0]),
                            dims=("row"),
                            coords={"ROWID":("row",rowid)
                                    })
    })
    
    return maintable

    
def open_dataset(zarr_path:str,column:str='COMPRESSED_DATA',group:str=None):
    """"
    Open the zarr store in a MSv2 format including the SVD components.
    
    Parameters
    ------
    zarr path (str)
        The path to the zarr store.
    
    column (str)
        The column in which the compressed data is stored in in the zarr store.
    
    group (str)
        MS group/subtable to open. If none, the main table is opened. Default is None.
    
    Returns
    -----
    dataset (xarray dataset)
    """
    if group == None:
        maintable = construct_main_ds(zarr_path=zarr_path,column=column)
        return maintable
    else:
        ds = xr.open_zarr(zarr_path,group=group,consolidated=True)
        
        return ds
    
        
def write_datasets_to_ms(zarr_path:str,msname:str,column:str):
    """"
    Write all the datasets to the Measurement set.
    
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
    if os.path.exists(msname):
        shutil.rmtree(msname)
       
    maintable =  construct_main_ds(zarr_path=zarr_path,column=column)    
    write_main = xds_to_table(maintable,f"{msname}")
    
    from dask.diagnostics import ProgressBar
    with ProgressBar():
        dask.compute(write_main) 
    
    
    zarr_folders  = list_subtables(zarr_path)
    non_folders = ['MAIN','FLAGS','FLAG_ROW','WEIGHT_SPECTRUM']
    tasks = []
    for folder in zarr_folders:
        if folder in non_folders:
            continue
        task = write_subtable(zarr_path,msname,folder)
        tasks.append(task)
        
    dask.compute(*tasks)
      