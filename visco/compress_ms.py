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
import dask
from tqdm.dask import TqdmCallback
import visco
log = visco.get_logger(name="VISCO")
import logging
from omegaconf import OmegaConf
logging.getLogger("daskms").setLevel(logging.ERROR)
logging.getLogger('numcodecs').setLevel(logging.CRITICAL)

from numcodecs import Blosc, Zstd, GZip

CORR_TYPES = OmegaConf.load(f"{visco.PCKGDIR}/ms_corr_types.yaml").CORR_TYPES
CORR_TYPES_REVERSE = OmegaConf.load(f"{visco.PCKGDIR}/ms_corr_types_reverse.yaml").CORR_TYPES


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
            flags:da.Array,
            use_model_data:bool=None,
            model_data:da.Array=None,
            flagvalue:float=None,
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
    use_model_data : bool, optional
        Whether to use model data to replace the flags.
    model_data : dask.array.Array, optional
        Model data to use if `use_model_data` is True.
    flagvalue : float, optional
        Value to replace flagged data with.
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
    
    if use_model_data:
        visdata = visdata.where(visdata == flags,model_data,visdata)
    if flagvalue:
        visdata = da.where(visdata == flags, flagvalue, visdata)
    
    U,S,Vt = da.linalg.svd(visdata)
    
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



def compress_visdata(zarr_output_path:str,
                     correlation:str,
                     fieldid:int,
                     ddid:int,
                     scan:int,
                     column:str,
                     use_model_data:bool,
                     decorrelation:float=None,
                     compressionrank:int=None, 
                     flagvalue:int=None, 
                     antennas:list=None, 
                     ):
    """
    Compress visibility data using SVD.

    Parameters
    ----------
    visdata : dask.array.Array
        Visibility data to compress.
    decorrelation : float, optional
        Desired decorrelation level (0 to 1). Default is None.
    compressionrank : int, optional
        Number of singular values to keep. Default is None.

    Returns
    -------
    dask.array.Array
        Compressed visibility data.
    """
    
    
    ds = xr.open_zarr(zarr_output_path, consolidated=True)
    ds_pol = xr.open(zarr_output_path, consolidated=True, group='POLARIZATION')
    ds_ant = xr.open_zarr(zarr_output_path, consolidated=True, group='ANTENNA')
    
    scans = da.unique(ds.SCAN_NUMBER.values)
    ddids = da.unique(ds.DATA_DESC_ID.values)
    fields = da.unique(ds.FIELD_ID.values)
    corr_types = ds_pol.CORR_TYPE.values
    
    if scan not in scans:
        raise ValueError(f"Invalid SCAN_NUMBER {scan}. Available scans are: {scans.compute().tolist()}")
        
    
    if ddid not in ddids:
        raise ValueError(f"Invalid selected DATA_DESC_ID {ddid}.\
                            Available DATA_DESC_IDs are {ddids}")
    
    
    if fieldid not in fields:
        raise ValueError(f"Invalid selected FIELD_ID {fieldid}. Available FIELD_ID's are\
                            {fields}.")
    
    maintable = ds.where(
        (ds.SCAN_NUMBER == scan) & 
        (ds.DATA_DESC_ID == ddid) & 
        (ds.FIELD_ID == fieldid), 
        drop=True
    )
    
    ant1 = maintable.ANTENNA1.values
    ant2 = maintable.ANTENNA2.values
    
    if antennas:
        if isinstance(antlist, str): 
            try:
                antlist = ast.literal_eval(antlist)
                if not isinstance(antlist, list) or not all(isinstance(x, int) for x in antlist):
                    raise ValueError("Parsed antlist is not a valid list of integers.")
            except (ValueError, SyntaxError):
                raise ValueError(f"Invalid format for antlist: {antlist}. Expected a list of integers.")
        
        baselines = list(combinations(antlist, 2))      
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
        
    
    tasks = []
    
    for bx,(antenna1,antenna2) in enumerate(baselines):
        
        baseline_mask = (maintable.ANTENNA1 == antenna1) & (maintable.ANTENNA2 == antenna2)
        baseline_data = maintable.where(baseline_mask, drop=True)

        
        
        ant1name = ds_ant.NAME.values[antenna1]
        ant2name = ds_ant.NAME.values[antenna2]
    
        
        #Go through the given correlations
        for c in corr_list_user:
            ci = np.where(corr_types == c)[0][0]
            flag = baseline_data.FLAG.data[:,:,ci]
            visdata = baseline_data[column].data[:,:,ci]
            
            if use_model_data:
                model_data = baseline_data.MODEL_DATA.data[:,:,ci]
            
            task = delayed(apply_svd)(
                visdata=visdata,
                flags=flag,
                use_model_data=use_model_data,
                model_data=model_data if use_model_data else None,
                flagvalue=flagvalue,
                decorrelation=decorrelation,
                compressionrank=compressionrank
            )
            
        
            corr_type = CORR_TYPES_REVERSE[c]
            
            save_path = Path(zarr_output_path) / f'baseline{bx}' / f'{ant1name}_{ant2name}' / f'{corr_type}'
            save_task = delayed(write_svd_to_zarr)(task, save_path)
            tasks.append(save_task)
            
    return task


def write_svd_to_zarr(svd_result, path: Path):
    U, s, V = svd_result
    
    # Ensure output group exists
    store = zarr.DirectoryStore(str(path))
    root = zarr.group(store=store, overwrite=True)
    
    root.create_dataset('U', data=U.compute(), chunks=True, compression='zstd')
    root.create_dataset('S', data=s.compute(), chunks=True, compression='zstd')
    root.create_dataset('V', data=V.compute(), chunks=True, compression='zstd')
    

def compress_ms(ms_path:str, zarr_path:str,
                consolidated:bool=True,
                chunk_size_row:int=100000,
                overwrite:bool=True,
                compressor:str="zstd",
                level:int=4,
                correlation:str='XX,YY',
                fieldid:int=0,
                ddid:int=0,
                scan:int=0,
                column:str='DATA',
                use_model_data:bool=True,
                decorrelation:float=None,
                compressionrank:int=None, 
                flagvalue:int=None, 
                antennas:list=None):
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
    fieldid : int
        Field ID to filter by. Default is 0.
    ddid : int
        Data description ID to filter by. Default is 0.
    scan : int
        Scan number to filter by. Default is 0.
    column : str
        Column name for visibility data (e.g., 'DATA'). Default is 'DATA'.
    use_model_data : bool
        Whether to use model data for compression. Default is True.
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
    
    zarr_output_path = os.path.join(os.getcwd(), "compression-output", f"{zarr_path}")
    write_ms_to_zarr(ms_path=ms_path,
                    zarr_path=zarr_output_path,
                    consolidated=consolidated,
                    chunk_size_row=chunk_size_row,
                    overwrite=overwrite,
                    compressor=compressor,
                    level=level)
    
    tasks = delayed(compress_visdata)(ms_path=ms_path,
                                      zarr_path=zarr_output_path,
                                      correlation=correlation,
                                      fieldid=fieldid,)
    

             
  
  
    

    
    
    