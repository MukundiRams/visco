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
from daskms import xds_to_table
import visco
log = visco.get_logger(name="VISCO")
from tqdm import tqdm
import sys

class UnifiedProgressBar:
    def __init__(self):
        self.pbar = None
        self.total_steps = 0
        self.current_step = 0
        
    def start_progress(self, total_steps, initial_desc="Starting..."):
        """Start the master progress bar with total number of steps"""
        if self.pbar is not None:
            self.pbar.close()
        self.total_steps = total_steps
        self.current_step = 0
        self.pbar = tqdm(total=total_steps, desc=initial_desc, file=sys.stdout,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}')
        
    def update_step(self, description=None):
        """Move to next step and update description"""
        self.current_step += 1
        if self.pbar is not None:
            self.pbar.update(1)
            if description:
                self.pbar.set_description(description)
                
    def close(self):
        """Close the progress bar"""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


progress = UnifiedProgressBar()

def write_subtable(zarr_path: str, msname: str, group: str):
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

    ds = xr.open_zarr(zarr_path, group=group)
    
    if "ROWID" in ds.coords:
        ds = ds.reset_coords("ROWID", drop=True)
        
    write = xds_to_table(ds, f"{msname}::{group}")
    
    return write

def list_subtables(zarr_path: str):
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
    
def unstack_vis(vis_reconstructed, nrows):
    """Return list of blocks each with shape (nrows, nchan)."""
    if isinstance(vis_reconstructed, da.Array):
        nstack = int(vis_reconstructed.shape[0] // nrows)
        return [vis_reconstructed[i*nrows:(i+1)*nrows, :] for i in range(nstack)]
    else:
        nstack = vis_reconstructed.shape[0] // nrows
        return list(np.split(vis_reconstructed, nstack, axis=0))

def reconstruct_vis(U: da.Array, S: da.Array, Vt: da.Array) -> da.Array:
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
    
    if S.ndim == 2:
        S = S[:, 0]  
    
    S_reshaped = S.reshape((1, S.shape[0]))
    
    U_mult_S = U * S_reshaped
    
    return U_mult_S @ Vt

def construct_main_ds(zarr_path: str, column: str):
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
    
    progress.update_step("Loading main table and antenna data")
    maintable = xr.open_zarr(zarr_path, group='MAIN', consolidated=True)
    antennas = xr.open_zarr(zarr_path, group='ANTENNA', consolidated=True)
    antnames = antennas.NAME.values

    ant1 = maintable.ANTENNA1.values
    ant2 = maintable.ANTENNA2.values
    data_shape = maintable.DATA.shape  
    rowid = maintable.coords['ROWID'].values
    chunks = maintable.DATA.chunks

    reconstructed_data = da.zeros(data_shape, dtype=maintable.DATA.dtype, chunks=chunks)
    
    baselines = list_subtables(f"{zarr_path}/MAIN/{column}")
    
   
    for baseline_idx, baseline in enumerate(baselines):
        progress.update_step(f"Reconstructing baseline {baseline_idx+1}/{len(baselines)}")
        
        correlations = list_subtables(f"{zarr_path}/MAIN/{column}/{baseline}")
        ant1_name, ant2_name = baseline.split('&')
        try:
            ant1_idx = np.where(antnames == ant1_name)[0][0]
            ant2_idx = np.where(antnames == ant2_name)[0][0]
        except IndexError:
            log.warning(f"Baseline {baseline} not found in ANTENNA table. Skipping.")
            continue
        
        baseline_mask = (ant1 == ant1_idx) & (ant2 == ant2_idx)
        row_indices = np.where(baseline_mask)[0]
        nrows = row_indices.size
        corr_indices = {'XX': 0, 'XY': 1, 'YX': 2, 'YY': -1}
        
        if not correlations:
            continue
            
        tasks = []
        for corr_name in correlations:
            components = xr.open_zarr(f"{zarr_path}/MAIN/{column}/{baseline}/{corr_name}")
            U = components.U.data  
            S = components.S.data
            Vt = components.WT.data

            vis_reconstructed = delayed(reconstruct_vis)(U, S, Vt)
            tasks.append(vis_reconstructed)
        
        vis_reconstructed_list = dask.compute(*tasks)
        
        for i, corr_name in enumerate(correlations):
            vis_reconstructed = vis_reconstructed_list[i]
            if corr_name == 'diagonals':
                parts = unstack_vis(vis_reconstructed, nrows)
                reconstructed_data[row_indices, :, 0] = parts[0]
                reconstructed_data[row_indices, :, 3] = parts[1]
            elif corr_name == 'offdiagonals':
                parts = unstack_vis(vis_reconstructed, nrows)
                reconstructed_data[row_indices, :, 1] = parts[0]
                reconstructed_data[row_indices, :, 2] = parts[1]
            else:
                corr_idx = corr_indices[corr_name]
                reconstructed_data[row_indices, :, corr_idx] = vis_reconstructed

    progress.update_step("Loading and processing flag data")
    
    flags_ds = xr.open_zarr(zarr_path, group='FLAGS', consolidated=True)
    flags_length = data_shape[0] * data_shape[1] * data_shape[2]
    flags = np.unpackbits(flags_ds.FLAGS.values, count=flags_length)
    flags = flags.reshape(data_shape)
    
    flag_row_ds = xr.open_zarr(zarr_path, group='FLAGS_ROW', consolidated=True)
    flags_row = np.unpackbits(flag_row_ds.FLAGS_ROW.values, count=data_shape[0])

    if 'WEIGHT_SPECTRUM' in list_subtables(f"{zarr_path}"):
        
        progress.update_step("Processing weight spectrum")
        
        weights = xr.open_zarr(f"{zarr_path}/WEIGHT_SPECTRUM", consolidated=True)
        weights_reconstructed = np.dot(weights.U.values, np.diag(weights.S.values))
        weights_expanded = np.expand_dims(weights_reconstructed, axis=-1)
        final_weights = np.tile(weights_expanded, (1, 1, data_shape[2]))
        
        maintable = maintable.assign(**{
            'WEIGHT_SPECTRUM': xr.DataArray(da.from_array(final_weights, chunks=chunks),
                                        dims=("row", "chan", "corr"),
                                        coords={"ROWID": ("row", rowid)}),
            'SIGMA_SPECTRUM': xr.DataArray(da.from_array(final_weights, chunks=chunks),
                                        dims=("row", "chan", "corr"),
                                        coords={"ROWID": ("row", rowid)})
        })
    
    progress.update_step("Finalizing main table dataset")
    maintable = maintable.assign(**{
        'DATA': xr.DataArray(reconstructed_data, 
                            dims=("row", "chan", "corr"),
                            coords={"ROWID": ("row", rowid)}),
        'FLAG': xr.DataArray(da.from_array(flags, chunks=chunks),
                            dims=("row", "chan", "corr"),
                            coords={"ROWID": ("row", rowid)}),
        'FLAG_ROW': xr.DataArray(da.from_array(flags_row, chunks=chunks[0]),
                            dims=("row"),
                            coords={"ROWID": ("row", rowid)})
    })
    
    return maintable

def open_dataset(zarr_path: str, column: str = 'COMPRESSED_DATA', group: str = None):
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
    if group is None:
        maintable = construct_main_ds(zarr_path=zarr_path, column=column)
        return maintable
    else:
        ds = xr.open_zarr(zarr_path, group=group, consolidated=True)
        return ds

def write_datasets_to_ms(zarr_path: str, msname: str, column: str):
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
    
    baselines = list_subtables(f"{zarr_path}/MAIN/{column}")
    zarr_folders = list_subtables(zarr_path)
    non_folders = ['MAIN', 'FLAGS', 'FLAG_ROW', 'WEIGHT_SPECTRUM']
    subtable_count = len([f for f in zarr_folders if f not in non_folders])
    
    #total steps: setup + baselines + flag processing + weight spectrum + finalize + write main + subtables + completion
    total_steps = 1 + len(baselines) + 3 + 1 + subtable_count + 1
    
    progress.start_progress(total_steps, "Starting MS decompression...")
    
    if os.path.exists(msname):
        progress.update_step("Removing existing MS")
        shutil.rmtree(msname)
    
    maintable = construct_main_ds(zarr_path=zarr_path, column=column)
    
    progress.update_step("Writing main table to Measurement Set")    
    write_main = xds_to_table(maintable, f"{msname}")
    

    progress.update_step("Computing main table (this may take a while)...")
    dask.compute(write_main)
  
    
    tasks = []
    for folder in zarr_folders:
        if folder in non_folders:
            continue
        progress.update_step(f"Writing subtable: {folder}")
        task = write_subtable(zarr_path, msname, folder)
        tasks.append(task)
        
    progress.update_step("Finalizing Measurement Set")
    dask.compute(*tasks)
    
    progress.update_step("Measurement Set creation completed successfully")
    progress.close()
    