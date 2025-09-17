import logging
logging.getLogger('numcodecs').setLevel(logging.CRITICAL)
logging.getLogger("daskms").setLevel(logging.ERROR)


import dask.array as da
import numpy as np
import xarray as xr
import os
import shutil
from pathlib import Path
import zarr
import ast
from itertools import combinations
from daskms import xds_from_table
from dask import delayed
from scipy.interpolate import griddata
import dask
from tqdm import tqdm
from tqdm.dask import TqdmCallback
import visco
log = visco.get_logger(name="VISCO")
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="daskms")

from visco import setup_dask_client

import numcodecs
from numcodecs import Blosc, Zstd, GZip




CORR_TYPES = OmegaConf.load(f"{visco.PCKGDIR}/ms_corr_types.yaml").CORR_TYPES
CORR_TYPES_REVERSE = OmegaConf.load(f"{visco.PCKGDIR}/ms_corr_types_reverse.yaml").CORR_TYPES

_global_progress = None


def get_compressor(name:str=None, level:int=None):
    if name.lower() == "zstd":
        return Zstd(level=level)
    elif name.lower() == "gzip":
        return GZip(level=level)
    elif name.lower() == "blosc":
        return Blosc(cname="lz4", clevel=level)
    elif name is None:
        return None
    else:
        raise ValueError(f"Unsupported compressor: {name}")

def write_table_to_zarr(ms_path:str, zarr_path:str, 
                        consolidated:bool=None, 
                        chunk_size_row:int=None,
                        subtable:str=None,
                        overwrite:bool=None, 
                        compressor:str=None, 
                        level:int=None):
    """
    Convert a Measurement Set table to a Zarr store.

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set directory (e.g., 'mydata.ms').
    zarr_path : str
        Path to the output Zarr store directory (e.g., 'mydata.zarr').
    consolidated : bool
        Whether to write consolidated metadata for faster loading. Default is True.
    chunk_size_row : int
        Chunk size for the 'row' dimension in the main table and large subtables. Default is 100000.

    Returns
    -------
    None
    """
    
    if _global_progress:
        _global_progress.set_description(f"Converting MS to a Zarr store")
    
    if subtable:
        table_path = f"{ms_path}/{subtable}"
    else:
        table_path = ms_path
        
    if os.path.exists(zarr_path) and overwrite:
        shutil.rmtree(zarr_path)
        
        
    dataset = xds_from_table(table_path)
    
    codec = get_compressor(compressor, level)
    
    writes = []

    for i, ds in enumerate(dataset):
        if subtable:
            group_name = subtable
        else:
            group_name = 'MAIN'
            
        if chunk_size_row:
            ds = ds.chunk({'row': chunk_size_row})

        encoding = {
            var: {
                "compressor": codec,
                "chunks": (chunk_size_row,) + ds[var].shape[1:] if chunk_size_row else ds[var].chunks
            }
            for var in ds.data_vars
        }

        writes.append(
            ds.to_zarr(
                zarr_path,
                consolidated=consolidated,
                group=group_name,
                compute=False,
                mode='w' if overwrite and i == 0 else 'a',
                encoding=encoding
            )
        )
    
    dask.compute(*writes)
    
    if _global_progress:
        _global_progress.update(1)

   
        
def write_ms_to_zarr(ms_path:str, zarr_path:str,
                    consolidated:bool, 
                    chunk_size_row:int,
                    overwrite:bool, 
                    compressor:str, 
                    level:int):
    """
    Convert a Measurement Set to a Zarr store.

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set directory (e.g., 'mydata.ms').
    zarr_path : str
        Path to the output Zarr store directory (e.g., 'mydata.zarr').

    Returns
    -------
    None
    """

    write_table_to_zarr(
        ms_path = ms_path,
        zarr_path= zarr_path,
        consolidated=consolidated,
        chunk_size_row=chunk_size_row,
        overwrite=overwrite,
        compressor=compressor,
        level=level
    )
    
    def list_subtables(ms_path):
        return [f for f in os.listdir(ms_path) if os.path.isdir(os.path.join(ms_path, f))]
    
    subtables = list_subtables(ms_path)
    
    for subtable in subtables:
        write_table_to_zarr(ms_path=ms_path,
                            zarr_path=zarr_path,
                            consolidated=consolidated,
                            chunk_size_row=chunk_size_row,
                            subtable=subtable,
                            overwrite=False,
                            compressor=compressor,
                            level=level
                            )
        

def find_n_decorrelation(singular_values:np.ndarray, decorrelation:float)->int:
    """
    Find the number of singular values needed to reach a specified decorrelation level.

    Parameters
    ----------
    singular_values : Array
        Array of singular values.
    decorrelation : float
        Desired decorrelation level.

    Returns
    -------
    int
        Number of singular values needed to reach the decorrelation level.
    """
    
    sum_total = da.sum(singular_values**2).compute()
    threshold = (decorrelation)**2 * sum_total
    cumulative = da.cumsum(singular_values**2).compute()
    
    n = np.argmax(cumulative >= threshold) + 1  #+1 to convert index to count
    if n == 0:
        n = len(singular_values)
    
    return n

def apply_svd(visdata:da.Array, 
            decorrelation:float=None,
            compressionrank:int=None):
    """
    Decompose a baseline using SVD.

    Parameters
    ----------
    visdata : dask.array.Array
        Visibility data for the baseline.
    flags : dask.array.Array
        Flags for the visibility data.
    decorrelation : float, optional
        Desired decorrelation level (0 to 1).
    compressionrank : int, optional
        Number of singular values to keep.

    Returns
    -------
    U : dask.array.Array
        Left singular vectors.
    S : dask.array.Array
        Singular values.
    Vt : dask.array.Array
        Right singular vectors.
    """
    
    
    U,S,Vt = da.linalg.svd(da.from_array(visdata))
    
    if compressionrank:
        n = compressionrank
        
    elif decorrelation:
        n = find_n_decorrelation(S, decorrelation)
        
    else:
        n = len(S)
    
    U = U[:,:n]
    S = S[:n]
    Vt = Vt[:n,:]
    
    return U, S, Vt




def estimate_flagged_data(maintable: xr.Dataset) -> xr.Dataset:
    """
    Estimate the values of the flagged data in the visibilities by interpolating
    over the uv-plane.

    Parameters
    ----------
    maintable : xr.Dataset
        Main xarray dataset containing visibilities, flags, UVW, and weights.

    Returns
    -------
    xr.Dataset
        Updated dataset with flagged values estimated and filled.
    """
    
    if _global_progress:
        _global_progress.set_description("Estimating flagged data (this may take time)")
    
    required_fields = ['FLAG', 'DATA', 'UVW', 'WEIGHT_SPECTRUM']
    
    if not all(key in maintable for key in required_fields):
        raise ValueError(f"Input dataset missing required fields: {required_fields}")

    
    flags = maintable.FLAG.data          
    visdata = maintable.DATA.data        
    uvw = maintable.UVW.data.compute()   
    weights = maintable.WEIGHT_SPECTRUM.data  

    u = uvw[:, 0]
    v = uvw[:, 1]

    def interpolate_and_replace(vis, flag, weight, block_info=None):
        
        #Determine the slice range for this block
        loc = block_info[0]['array-location'][0]
        start, stop = loc
        u_block = u[start:stop]
        v_block = v[start:stop]

        valid = ~flag
        if valid.sum() < 3:
            return vis

        
        u_valid = u_block[valid]
        v_valid = v_block[valid]
        vis_valid = vis[valid]
        w_valid = weight[valid]

        
        flagged_uv = np.stack([u_block[flag], v_block[flag]], axis=-1)
        interpolated = griddata(
            (u_valid, v_valid),
            vis_valid,
            flagged_uv,
            method='linear',
            fill_value=0.0
        )

        
        vis_out = vis.copy()
        vis_out[flag] = interpolated
        return vis_out

    
    def process_slice(vis_slice, flag_slice, weight_slice):
        return da.map_blocks(
            interpolate_and_replace,
            vis_slice,
            flag_slice,
            weight_slice,
            dtype=vis_slice.dtype,
            block_info=True
        )

    nchan = visdata.shape[1]
    npol = visdata.shape[2]

    updated_visdata = da.stack([
        da.stack([
            process_slice(visdata[:, chan, pol], flags[:, chan, pol], weights[:, chan, pol])
            for pol in range(npol)
        ], axis=-1)
        for chan in range(nchan)
    ], axis=1)

    updated_visdata = da.rechunk(updated_visdata,chunks=visdata.chunks)
    # updated_ds = maintable.copy()
    # updated_ds['DATA'].data = da.rechunk(updated_visdata,chunks=visdata.chunks)
    return updated_visdata
  
def write_a_group_to_zarr(zarr_path:str,group:str,data:np.ndarray):
    """
    Write (updates) a group in the zarr store.
    """
    ds = xr.Dataset({
        group:(("row"), data)
        },
        coords={
            "row": np.arange(data.shape[0])
        }
    )
    ds.to_zarr(zarr_path,group=f"{group}",mode='a')


def compress_visdata(zarr_output_path:str,
                     compressor:str,
                     level:int,
                     correlation:str,
                     correlation_optimized:bool,
                     fieldid:int,
                     ddid:int,
                     scan:int,
                     column:str,
                     outcolumn:str,
                     flag_estimate:bool,
                     use_model_data:bool,  
                     model_data:da.Array=None,
                     decorrelation:float=None,
                     compressionrank:int=None, 
                     flagvalue:int=None,
                     antennas:list=None, 
                     ):
    """
    Compress visibility data using SVD.

    Parameters
    ----------
    decorrelation : float, optional
        Desired decorrelation level (0 to 1). Default is None.
    compressionrank : int, optional
        Number of singular values to keep. Default is None.
    use_model_data : bool, optional
        Whether to use model data to replace the flags.
    model_data : dask.array.Array, optional
        Model data to use if `use_model_data` is True.
    flagvalue : float, optional
        Value to replace flagged data with.

    Returns
    -------
    dask.array.Array
        Compressed visibility data.
    """
    
    ds = xr.open_zarr(zarr_output_path, consolidated=True,group='MAIN')
    ds_pol = xr.open_zarr(zarr_output_path, consolidated=True, group='POLARIZATION')
    ds_ant = xr.open_zarr(zarr_output_path, consolidated=True, group='ANTENNA')
    
    scans = np.unique(ds.SCAN_NUMBER.values)
    ddids = np.unique(ds.DATA_DESC_ID.values)
    fields = np.unique(ds.FIELD_ID.values)
    corr_types = ds_pol.CORR_TYPE.values
    
    if scan not in scans:
        raise ValueError(f"Invalid SCAN_NUMBER {scan}. Available scans are: {scans.tolist()}")
        
    
    if ddid not in ddids:
        raise ValueError(f"Invalid selected DATA_DESC_ID {ddid}.\
                            Available DATA_DESC_IDs are {ddids}")
    
    
    if fieldid not in fields:
        raise ValueError(f"Invalid selected FIELD_ID {fieldid}. Available FIELD_ID's are\
                            {fields}.")
    
    maintable = ds.where(
        (ds.SCAN_NUMBER == scan) & 
        (ds.DATA_DESC_ID == ddid) & 
        (ds.FIELD_ID == fieldid)
    )
    
    if _global_progress:
        _global_progress.set_description("Processing flags")
        
    flags = maintable.FLAG.astype(bool).values
    flags_row = maintable.FLAG_ROW.astype(bool).values
    
    flags_packed = np.packbits(flags, axis=None)
    flags_row_packed = np.packbits(flags_row, axis=None)
    
    write_a_group_to_zarr(zarr_output_path,'FLAGS_ROW',flags_row_packed)
    write_a_group_to_zarr(zarr_output_path,'FLAGS',flags_packed)
    
    
    if "WEIGHT_SPECTRUM" in maintable.data_vars:
        if _global_progress:
            _global_progress.set_description("Compressing WEIGHT SPECTRUM")
        
        ws = maintable.WEIGHT_SPECTRUM.values[:,:,0]
        wscomps = apply_svd(ws,compressionrank=1)
        ws_path = Path(zarr_output_path) / 'WEIGHT_SPECTRUM'
        write_svd_to_zarr(wscomps,ws_path,compressor,level,maintable.coords['ROWID'].values)
        delete_zarr_groups(zarr_output_path,f"MAIN/WEIGHT_SPECTRUM")
        delete_zarr_groups(zarr_output_path,f"MAIN/SIGMA_SPECTRUM")
    
    
    ant1 = maintable.ANTENNA1.values
    ant2 = maintable.ANTENNA2.values
    
    if antennas:
        if isinstance(antennas, str): 
            try:
                antennas = ast.literal_eval(antennas)
                if not isinstance(antennas, list) or not all(isinstance(x, int) for x in antennas):
                    raise ValueError("Parsed antennas is not a valid list of integers.")
            except (ValueError, SyntaxError):
                raise ValueError(f"Invalid format for antennas: {antennas}. Expected a list of integers.")
        
        baselines = list(combinations(antennas, 2))      
    else:
        baselines = np.unique([(a1, a2) for a1, a2 in zip(ant1,ant2) if a1 != a2],axis=0)
    
    corr_list = []
    for corr in corr_types:
        for cor in corr:
            corr_name = CORR_TYPES_REVERSE[int(cor)]
            corr_list.append(corr_name)
    
    corr_list_user = []
    for corr in correlation.split(','):
        corr_ind = CORR_TYPES[str(corr)]
        corr_list_user.append(corr_ind)
        
    maintable_copy = maintable.copy()
    
    if use_model_data:
        
        if _global_progress:
            _global_progress.set_description("Replacing flagged data with model")
        
        if model_data is None:
            mod_data = maintable.MODEL_DATA.data
        else:
            mod_data = maintable[model_data].data
        
        maintable_copy[column].data = da.where(maintable_copy[column].data == maintable_copy.FLAG.data, \
            mod_data, maintable_copy[column].data)
        
    elif flag_estimate:
        log.warning(f"Using this method may significantly increase the computational time.\
            This method uses interpolation to estimate the values of the flagged data.")
        
        updated_vis = estimate_flagged_data(maintable)
        maintable_copy[column].data = updated_vis
        
    elif flagvalue:
        
        if _global_progress:
            _global_progress.set_description(f"Replacing flagged data with {flagvalue}")
        
        log.warning(f"Using this flag replacement method may lead to the amplification of noise,\
            which might significantly affect the SVD compression. This is not recommended.")
        maintable_copy[column].data = da.where(maintable_copy[column].data == maintable_copy.FLAG.data, \
            flagvalue, maintable_copy[column].data)
    
    else:
        log.warning("No flag replacement method specified. If there are flagged data, they will not be replaced.")
    
    
    
    tasks = []
    if _global_progress:
        _global_progress.set_description("Processing baselines for SVD compression")
        
    processed_baselines = 0
    
    for bx,(antenna1,antenna2) in enumerate(baselines):
        
        antenna1 = int(antenna1)
        antenna2 = int(antenna2)
        baseline_mask = (ant1 == antenna1) & (ant2 == antenna2)
        baseline_data = maintable_copy.isel(row = baseline_mask)
        row_idx = baseline_data.coords['ROWID'].data

        
        ant1name = ds_ant.NAME.values[antenna1]
        ant2name = ds_ant.NAME.values[antenna2]
        
        if correlation_optimized:
            if 'XX' and 'YY' in correlation.split(","):
                cixx = np.where(corr_types[0] == 9)[0][0]
                ciyy = np.where(corr_types[0] == 12)[0][0]
                diag_visdata = np.vstack((baseline_data[column].data[:,:,cixx],baseline_data[column].data[:,:,ciyy]))
                
                task = delayed(apply_svd)(
                    visdata=diag_visdata,
                    decorrelation=decorrelation,
                    compressionrank=compressionrank
                )
                
                diag_row_idx = np.tile(row_idx, reps=2)
                save_path = Path(zarr_output_path) / 'MAIN'/ f"{outcolumn}" / f'{ant1name}&{ant2name}' / 'diagonals'
                save_task = delayed(write_svd_to_zarr)(task, save_path,compressor,level,diag_row_idx)
                tasks.append(save_task)
                
                processed_baselines += 1
                if _global_progress:
                    _global_progress.update(1)
            
            if 'XY' and 'YX' in correlation.split(","):
                cixy = np.where(corr_types[0] == 10)[0][0]
                ciyx = np.where(corr_types[0] == 11)[0][0]
                offdiag_visdata = np.vstack((baseline_data[column].data[:,:,cixy],baseline_data[column].data[:,:,ciyx]))
                
                task = delayed(apply_svd)(
                    visdata=offdiag_visdata,
                    decorrelation=decorrelation,
                    compressionrank=compressionrank
                )
                
                off_diag_row_idx = np.tile(row_idx, reps=2)
                save_path = Path(zarr_output_path) / 'MAIN'/ f"{outcolumn}" / f'{ant1name}&{ant2name}' / 'offdiagonals'
                save_task = delayed(write_svd_to_zarr)(task, save_path,compressor,level,off_diag_row_idx)
                tasks.append(save_task)
                
                processed_baselines += 1
                if _global_progress:
                    _global_progress.update(1) 
            
        else:    
            #Go through the given correlations
            for c in corr_list_user:
                ci = np.where(corr_types[0] == c)[0][0]
                visdata = baseline_data[column].data[:,:,ci]
                
                task = delayed(apply_svd)(
                    visdata=visdata,
                    decorrelation=decorrelation,
                    compressionrank=compressionrank
                )
                
                corr_type = CORR_TYPES_REVERSE[c]
                
                save_path = Path(zarr_output_path) / 'MAIN'/ f"{outcolumn}" / f'{ant1name}&{ant2name}' / f'{corr_type}'
                save_task = delayed(write_svd_to_zarr)(task, save_path,compressor,level,row_idx)
                tasks.append(save_task)
                
                # baseline_progress.update(1) 
                processed_baselines += 1
                if _global_progress:
                    _global_progress.update(1)
         
    return tasks


def write_svd_to_zarr(svd_result, path: Path,compressor:str,level:int,rowid:np.ndarray):
    
    U, s, V = svd_result
    
    
    store = zarr.DirectoryStore(str(path))
    root = zarr.group(store=store, overwrite=True)
    
    compressor = numcodecs.get_codec({'id': compressor, 'level': level})
    ds = xr.Dataset(
            {
            "U": (("time", "mode"), U.compute()),  
            "S": (("mode",), s.compute()),
            "WT": (("mode", "channel"), V.compute()),
            },
            coords={
            "time": rowid,
            "mode": np.arange(s.shape[0]),
            "channel": np.arange(V.shape[1]),
            })
    ds.to_zarr(store,mode='a',encoding={var: {"compressor": compressor} for var in ds.data_vars})



def compress_full_ms(ms_path:str, zarr_path:str,
                consolidated:bool,
                chunk_size_row:int,
                overwrite:bool,
                compressor:str,
                level:int,
                nworkers:int,
                nthreads:int,
                memory_limit:str,
                direct_to_workers:bool,
                silence_logs:str,
                correlation:str,
                correlation_optimized:bool,
                fieldid:int,
                ddid:int,
                scan:int,
                column:str,
                outcolumn:str,
                dashboard_addr:str=None,
                use_model_data:bool=False,
                model_data:str=None,
                flag_estimate:bool=False,
                decorrelation:float=None,
                compressionrank:int=None, 
                flagvalue:int=None, 
                antennas:list=None,
                ):
    """
    Compress a Measurement Set using SVD and save to Zarr.

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set directory.
    zarr_path : str
        Path to the output Zarr store directory.
    consolidated : bool
        Whether to write consolidated metadata for faster loading. Default is True.
    chunk_size_row : int
        Chunk size for the 'row' dimension in the main table and large subtables. Default is 100000.
    overwrite : bool
        Whether to overwrite existing Zarr store. Default is True.
    compressor : str
        Compression algorithm to use ('zstd', 'gzip', 'blosc'). Default is 'zstd'.
    level : int
        Compression level (1-9). Default is 4.
    correlation : str
        Correlation types to compress (e.g., 'XX,YY'). Default is 'XX,YY'.
    correlation_optimized : bool
        Whether to optimize the SVD compression by compressing XX and YY together, XY and YX together, reducing the computational time.
    fieldid : int
        Field ID to filter by. Default is 0.
    ddid : int
        Data description ID to filter by. Default is 0.
    scan : int
        Scan number to filter by. Default is 0.
    column : str
        Column name for visibility data (e.g., 'DATA'). Default is 'DATA'.
    outcolumn: str
        Output column name for compressed data. Default is 'COMPRESSED_DATA'.
    use_model_data : bool
        Whether to use model data for compression. Default is True.
    model_data: str
        The column which contains the model data to use to estimate the flagged values.
    decorrelation : float
        Desired decorrelation level (0 to 1). Default is None.
    compressionrank : int
        Number of singular values to keep. Default is None.
    flagvalue : int
        Value to replace flagged data with. Default is None.
    antennas : list
        List of antenna indices to compress. If None, all baselines are used.

    Returns
    -------
    """
    global _global_progress
    
    if not os.path.exists(ms_path):
        raise ValueError(f"Measurement Set path does not exist: {ms_path}")

    client = setup_dask_client(memory_limit=memory_limit,nworkers=nworkers,nthreads=nthreads,
                               direct_to_workers=direct_to_workers,
                               silence_logs=silence_logs, dashboard_addr=dashboard_addr)

    work_breakdown = calculate_total_work(ms_path, correlation, correlation_optimized, antennas)
    total_work = sum(work_breakdown.values())

    _global_progress = tqdm(total=total_work, desc="MS Compression Pipeline.", unit="tasks")
    
    
    zarr_output_path = os.path.join(zarr_path)
    write_ms_to_zarr(ms_path=ms_path,
                    zarr_path=zarr_output_path,
                    consolidated=consolidated,
                    chunk_size_row=chunk_size_row,
                    overwrite=overwrite,
                    compressor=compressor,
                    level=level)
    
    _global_progress.set_description("Processing visibility data")
    tasks = compress_visdata(
                            zarr_output_path=zarr_output_path,
                            compressor=compressor,
                            level=level,
                            correlation=correlation,
                            correlation_optimized=correlation_optimized,
                            fieldid=fieldid,
                            ddid=ddid,
                            scan=scan,
                            column=column,
                            outcolumn=outcolumn,
                            use_model_data=use_model_data,
                            model_data=model_data,
                            flag_estimate=flag_estimate,
                            decorrelation=decorrelation,
                            compressionrank=compressionrank,
                            flagvalue=flagvalue,
                            antennas=antennas)
    
    _global_progress.set_description("Computing SVD compression")
    with TqdmCallback(desc=f"Writing the final data"):
        dask.compute(*tasks)
    
    
    _global_progress.update(work_breakdown['final_compute'])
        
    _global_progress.set_description("Finalizing and cleaning up")
    delete_zarr_groups(zarr_output_path,"MAIN/FLAG")
    delete_zarr_groups(zarr_output_path,"MAIN/FLAG_ROW")
    delete_zarr_groups(zarr_output_path,f"MAIN/{column}")
    
    
    if use_model_data:
        delete_zarr_groups(zarr_output_path,f"MAIN/{model_data}")
    
    _global_progress.update(work_breakdown['cleanup'])
    _global_progress.set_description("✅ MS compression completed successfully!")
    
    
    client.close()
    if _global_progress:
        _global_progress.close()
        _global_progress = None

def delete_zarr_groups(zarr_path:str,group:str):
    """
    Delete the specified group on the specified zarr store.
    
    Parameters
    ------
    zarr_path (str)
        The path to the zarr store.
    group (str)
        The group to delete.
    """

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)

    abs_path = os.path.join(store.path, group)

    if os.path.exists(abs_path):
        shutil.rmtree(abs_path)
        




def calculate_total_work(ms_path: str, correlation: str, correlation_optimized: bool, antennas: list = None):
    """Calculate the total amount of work to be done for accurate progress tracking."""
    
    def list_subtables(ms_path):
        return [f for f in os.listdir(ms_path) if os.path.isdir(os.path.join(ms_path, f))]
    
    subtables = list_subtables(ms_path)

    try:
        temp_ds = xds_from_table(ms_path)
        if temp_ds:
            sample_ds = temp_ds[0] 
            ant1 = sample_ds.ANTENNA1.values
            ant2 = sample_ds.ANTENNA2.values
            
            if antennas:
                baselines = list(combinations(antennas, 2))
            else:
                unique_baselines = set()
                for a1, a2 in zip(ant1, ant2):
                    if a1 != a2:
                        unique_baselines.add((min(a1, a2), max(a1, a2)))
                baselines = list(unique_baselines)
            
            corr_list_user = correlation.split(',')
            if correlation_optimized:
                baseline_work = len(baselines) * (len(corr_list_user)/2)
            else:
                baseline_work = len(baselines) * len(corr_list_user)
    except:
        #Fallback estimate
        baseline_work = 250
    
    return {
        'ms_to_zarr': 1 + len(subtables),  # Main table + subtables
        'flag_processing': 1,
        'weight_processing': 1,
        'baseline_processing': baseline_work,
        'final_compute': 1,
        'cleanup': 1
    }
    