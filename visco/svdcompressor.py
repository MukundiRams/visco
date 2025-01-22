import dask.array as da
from daskms import xds_from_table,xds_to_table
import numpy as np
from dask import delayed, compute
from typing import Union
from visco.msmodify import add_column,delete_column
from visco.msvolume import data_volume_calc
# from scabha.basetypes import File
from copy import deepcopy
from visco.utilities import ObjDict
import visco
log = visco.get_logger(name="visco")
from tqdm import tqdm
import logging
logging.getLogger("daskms").setLevel(logging.ERROR)



def apply_svd(dataset:da.Array, rank:Union[str,int]):
    
    """
    Decompose a dataset into three components, left singular vector U, descending
    order singular values S, and right singular vector U and reconstruct the dataset
    using the specified number of singular values.
    
    Parameters
    ----
        dataset: dask.Array 
            - Data matrix to compress using SVD.
    
        rank: Union[str,int] 
            - Rank reduction for the data matrix. Options are 'fullrank','halfrank'\
                or an integer. Default is 'fullrank'.
    
    Returns
    ----
        compressed_dataset: dask.Array
                - Compressed or reduced rank dataset.
        
        signal: float 
                - Amount of signal preserved by reducing the rank or compressing the data.
        
        loss: float 
                - Amount of signal lost by reducing the rank or compressing the data.
        
        Full_rank: int 
                - Full rank of the dataset.
        
        reduced_rank: int 
                - Used rank for reduction or compression.
        
        cf: float 
                - Compression factor on the data after reducing the rank.
    
    """
    
    original_shape = dataset.shape

    if not isinstance(dataset,da.Array):
        dataset = da.from_array(dataset)
        
    dataset = dataset.T
    m = dataset.shape[0]
    n = dataset.shape[1]
    #rank of the dataset is the minimum between the number of rows and channels
    full_rank  = min(m,n)
    
    #decomposing the dataset into three components
    u, s, v = da.linalg.svd(dataset)
    
    #rank options are 'fullrank', 'halfrank' and integers
    if rank == 'fullrank':
        reduced_rank = full_rank
    elif rank == 'halfrank':
        reduced_rank = full_rank // 2  
    else:
        reduced_rank = rank

    # reduced_s = s.copy()
    # reduced_s[rank:] = 0.0
    
    #simulate lossy compression by cutting of unwanted values
    reduced_s = s[:reduced_rank]
    reduced_u = u[:, :reduced_rank]
    reduced_v = v[:reduced_rank, :]
    
    #simulate lossy compression by cutting of unwanted values
    # reduced_u = u[:, reduced_rank].reshape(-1, 1)  
    # reduced_s = s[reduced_rank]                    
    # reduced_v = v[reduced_rank, :].reshape(1, -1)
    
    #reconstruct the dataset with the new rank, calculate the signal preserved and compression loss.
    compressed_dataset = reduced_u @ np.diag(reduced_s) @  reduced_v
    # compressed_dataset = reduced_u @ (reduced_s *  reduced_v)
    signal_preserved = da.linalg.norm(compressed_dataset, 'fro') / da.linalg.norm(dataset, 'fro')
    compression_loss = da.linalg.norm(dataset - compressed_dataset, 'fro') / da.linalg.norm(dataset, 'fro')

    
    #compression factor
    if m>0 and n>0:
        # assert reduced_rank>=1
        compression_factor = (m*n)/(reduced_rank*(m+n+0.5))
    else:
        compression_factor = 1
    
    compressed_dataset = compressed_dataset.T
    compressed_dataset = compressed_dataset.reshape(original_shape)
    
    result = ObjDict(
        {
            "compressed_dataset": compressed_dataset,
            "signal": signal_preserved,
            "loss":compression_loss,
            "cf":compression_factor,
            "full_rank": full_rank,
            "reduced_rank":reduced_rank
        }
    )
    
    return result


def get_baseline_info( msfile:str, chunks:dict, ant1:int, ant2:int):
    """
    Baseline information
    
    Parameters
    ----
        msfile: str 
            - Path to measurement set.
        
        chunks: dict 
            - Chunking of rows,channels and correlations. Default is\
            {'row':10000000, 'chan': 16, 'corr': 4}.
        
        ant1: int 
            - Index of antenna 1 for a baseline.
        
        ant2: int 
            - Index of antenna 2 for a baseline.
    
    Returns
    ----
        ds: dask.Array 
            - Data matrix for a baseline
    
    """
    
    ds = xds_from_table(msfile, chunks=chunks,
                        taql_where=f"ANTENNA1={ant1} AND ANTENNA2={ant2}")[0]
    return ds

def uniform_svd(msfile:str, chunks:dict = {'row':10000000, 'chan': 16, 'corr': 4},
                column:str = 'DATA', rank = 'fullrank', correlation:str = 'XX,XY,YX,YY',
                num_datacols:int=1, autocorr=False,outcol='COMPRESSED_DATA', 
                process='simulate', antlist:list=None,):
    
    """
    Compress visibility data using SVD for each baseline uniformly; using the\
        same number of singular values or rank
    
    Parameters
    ----
        msfile: Union[File,str] 
            - Path to measurement set.
        
        chunks: dict 
            - Chunking of rows,channels and correlations. Default is\
            {'row':10000000, 'chan': 16, 'corr': 4}.
        
        column: str 
            - Data column to be compressed. Default is DATA.
        
        rank: Union[str,int] 
            - Rank reduction for the data matrix. Options are 'fullrank','halfrank' or\
            an integer. Default is 'fullrank'. 
        
        correlation: str 
            - Selected correlations to perform SVD on. Default is 'XX,XY,YX,YY'.
        
        num_datacols: int 
            - Number of data columns in the MS.
        
        outcol: str
            - Column to store compressed data.
        
        autocorr: bool
            - Whether to include auto correlations in the compression.
        
        antlist: List
            - SVD can be performed on only specified baselines instead of the whole data\
            tensor by specifying antenna 1 and 2 indices in a list.
        
    
            
        
    Returns
    ----
        compressed_data: dask.Array 
            - Reduced rank data.
        
        signal: float 
            - Amount of signal preserved by reducing the rank or compressing the data.
        
        loss: float 
            - Amount of signal lost by reducing the rank or compressing the data.
        
        compression factor: float 
            - Compression factor on the data after reducing the rank.
        
        data size: float 
            - Data columns size in bytes.
        
        compressed data size: float 
            - Compressed data size in bytes.
    
    """
    
    corr = correlation.split(',') if isinstance(correlation, str) else corr
    corr_types = {'XX': 0, 'XY': 1, 'YX': 2, 'YY': -1}
    
    
    tab = xds_from_table(msfile, chunks=chunks)[0]
    
    
    #get all the antenna pairs making up all the baselines,
    if autocorr:
        unique_baselines = np.unique(list(zip(tab.ANTENNA1.data.compute(),\
                                              tab.ANTENNA2.data.compute())), axis=0)
    else:
        #filter out auto-correlations
        unique_baselines = np.unique(
            [(a1, a2) for a1, a2 in zip(tab.ANTENNA1.data.compute(),\
                tab.ANTENNA2.data.compute()) if a1 != a2], axis=0)
        
    nbaselines = len(unique_baselines)
    ncorr = len(corr)
    nrow = tab.DATA.data.shape[0]
    nchan = tab.DATA.data.shape[1]
    ntimes = nrow/nbaselines
    
    # if antlist:
    #     baselines = antlist
    # else:
    #     baselines = unique_baselines
        
    delayed_results = []
    # all_baselines_data = []
    
    log.info(f"Starting the process of compressing data column {column} with\
        {nbaselines} baselines and {ncorr} correlations uniformly.")
    
    #iterate over all the baselines
    for antenna1, antenna2 in tqdm(unique_baselines, desc="Processing baselines"):
        ds = get_baseline_info(msfile=msfile, chunks=chunks, ant1=antenna1, ant2=antenna2)
        
        #iterate over all the correlations
        for pol in corr:
            cid = corr_types[pol]
            data = ds[column].data[:, :, cid]
            output = delayed(apply_svd)(data, rank)
            delayed_results.append((output, pol))
              
    
    #process the delayed results
    computed_results = compute(*[delayed_result[0] for delayed_result in delayed_results],scheduler='threads', num_workers=64,optimize_graph=True)

    rec_matrix_dict = {pol: [] for pol in corr}
    cf = []

    for i, (result, pol) in enumerate(delayed_results):
        rec_matrix_dict[pol].append(computed_results[i].compressed_dataset)
        
        #only append compression ratio once per baseline
        if i % len(corr) == 0:  
            cf.append(computed_results[i].cf)
          
            
    fullrank = computed_results[0].full_rank
    reduced_rank = computed_results[0].reduced_rank
    log.info(f"The data has a full rank of {fullrank}. It was compressed to\
        reduced rank of {reduced_rank} ")
    
    log.info(f"Compression factor:{cf[0]}")
    
    #stack the reconstructed matrices for each correlation
    rec_matrix_stacked = {pol: da.stack(rec_matrix_dict[pol], axis=0) for pol in corr}
    
    #combine the stacked matrices along the correlation axis
    rec_matrix_combined = da.stack([rec_matrix_stacked[pol] for pol in corr], axis=-1)
    
    #reshape to match the original structure
    rec_matrix_combined = rec_matrix_combined.reshape((nbaselines,ntimes, nchan, ncorr))
    compressed_data = rec_matrix_combined.transpose(1,0, 2,3).reshape((ntimes * nbaselines, nchan, ncorr))

    add_column(msfile,outcol,compressed_data)
    if process == 'archive':
        delete_column(msfile,column)
        log.info(f"Compressed data has successfully been stored in data column {outcol},\
        and data column {column} has been deleted.")
    elif process == 'simulate':
        log.info(f"Compressed data has successfully been stored in data column {outcol}.")
    else:
        log.info(f"Data has successfully been compressed but not stored in the measurement set.")
   
    #compute data volumes
    total_data_volume = data_volume_calc(num_polarisations=ncorr,nrows=nrow,num_channels=nchan,
                                            num_data_cols=0)
    
    total_data_volume_with_data = data_volume_calc(num_polarisations=ncorr,nrows=nrow,
                                                   num_channels=nchan,
                                                num_data_cols=num_datacols)
    
    data_size = total_data_volume_with_data - total_data_volume
    data_corr_size = data_size / ncorr
    compressed_data_size = np.sum(np.ceil(data_corr_size * 1/cf[0]))
    
    log.info(f"The Measurement set has {num_datacols} data columns with a total volume\
            of {data_size} bytes, each data column has a volume of {data_size/num_datacols}\
            bytes.")
    
    log.info(f"The compressed data stored in {outcol} has a total volume of\
            {compressed_data_size} bytes.")

    results = ObjDict({
        "compressed_data": compressed_data,
        "cf": da.array(cf),
        "data_size": data_size,
        "compressed_data_size":compressed_data_size,
    })

    return results


def baseline_svd(msfile: str, chunks: dict = {'row': 10000000, 'chan': 16, 'corr': 4},
                 column: str = 'DATA', correlation: str = 'XX,XY,YX,YY', 
                 num_datacols: int = 1, autocorr: bool = False,
                 outcol: str = 'COMPRESSED_DATA', process: str = 'simulate',
                 decorrelation: float = None, cf: float = None):

    corr = correlation.split(',') if isinstance(correlation, str) else correlation
    corr_types = {'XX': 0, 'XY': 1, 'YX': 2, 'YY': -1}
    
    tab = xds_from_table(msfile, chunks=chunks)[0]
    
    if autocorr:
        unique_baselines = np.unique(list(zip(tab.ANTENNA1.data.compute(), tab.ANTENNA2.data.compute())), axis=0)
    else:
        unique_baselines = np.unique(
            [(a1, a2) for a1, a2 in zip(tab.ANTENNA1.data.compute(), tab.ANTENNA2.data.compute()) if a1 != a2],
            axis=0)
    
    nbaselines = len(unique_baselines)
    ncorr = len(corr)
    nrow = tab.DATA.data.shape[0]
    nchan = tab.DATA.data.shape[1]
    ntimes = nrow // nbaselines
    
    log.info(f"Starting the process of compressing data column {column} with {nbaselines} baselines and {ncorr} correlations baseline-dependently.")
    
    compressed_tensor = []
    signals = []
    cfs = []

    def process_correlation(data, corr_type, decorrelation, cf):
        compressed_data = None
        signal = None
        c_factor = None

        if decorrelation is not None:
            run = apply_svd(data, rank='fullrank')
            signal = run.signal
            rank = run.full_rank
            if signal > decorrelation:
                while signal > decorrelation and rank > 1:
                    rank -= 1
                    run = apply_svd(data, rank)
                    signal = run.signal
                    compressed_data = run.compressed_dataset
            elif signal == decorrelation:
                compressed_data = run.compressed_dataset
            else:
                log.debug(f"The provided minimum decorrelation is larger than the minimum\
                    decorrelation that can be achievedf.")
                
        elif cf is not None:
            run = apply_svd(data, rank='fullrank')
            c_factor = run.cf
            rank = run.full_rank
            if c_factor < cf:
                while c_factor < cf and rank > 1:
                    rank -= 1
                    run = apply_svd(data, rank)
                    c_factor = run.cf
                    compressed_data = run.compressed_dataset
            elif c_factor == cf:
                compressed_data = run.compressed_dataset
            else:
                log.debug(f"The provided compression factor is smaller than the minimum\
                    compression factor that can be achievedf.")
            
        
        return compressed_data, signal, c_factor

    tasks = []
    for antenna1, antenna2 in unique_baselines:
        ds = get_baseline_info(msfile=msfile, chunks=chunks, ant1=antenna1, ant2=antenna2)
        data_copy = deepcopy(ds[column].data)
        
        for pol in corr:
            cid = corr_types[pol]
            data = ds[column].data[:, :, cid]
            task = delayed(process_correlation)(data, pol, decorrelation, cf)
            tasks.append((task, data_copy, cid))

    # Compute the parallel tasks
    results = compute(*[task[0] for task in tasks])

    # Replace the data in the deep copy for each correlation
    for idx, (compressed_data, (task, data_copy, cid)) in enumerate(zip(results, tasks)):
        if compressed_data is not None:
            # print(f"Shape of data_copy[:, :, cid]: {data_copy[:, :, cid].shape}")
            # print(f"Shape of compressed_data: {compressed_data[0]}")
            data_copy[:, :, cid] = compressed_data[0]  # Update the copied data with the compressed correlation


    # After processing all baselines, reshape the final array
    compressed_tensor = da.stack([task[1] for task in tasks], axis=0)  # Stack the updated data copies
    compressed_tensor = compressed_tensor.reshape((nbaselines, ntimes, nchan, ncorr))
    compressed_result = compressed_tensor.transpose((1, 0, 2, 3)).reshape((ntimes * nbaselines, nchan, ncorr))

    # for result in results:
    #     compressed_data, signal, c_factor = result
    #     compressed_tensor.append(compressed_data)
    #     signals.append(signal)
    #     if c_factor is not None:
    #         cfs.append(c_factor)

    #  # Compute the Dask array before reshaping it using NumPy
    # compressed_tensor = da.stack(compressed_tensor, axis=0).compute()  # Use NumPy for reshaping
    # compressed_tensor = da.reshape(compressed_tensor, (nbaselines* ntimes, nchan, ncorr))
    
    # # The order of axes for Dask should be [times, baselines, channels, correlations]
    # # compressed_tensor = da.from_array(compressed_tensor, chunks=(ntimes, nbaselines, nchan, ncorr))
    
    # compressed_result = compressed_tensor.transpose((1, 0, 2, 3))  # Transpose to the correct shape
    # compressed_result = compressed_result.reshape((ntimes * nbaselines, nchan, ncorr))
    
    add_column(msfile, outcol, compressed_result)
    if process == 'archive':
        delete_column(msfile, column)
        log.info(f"Compressed data has successfully been stored in data column {outcol}, and data column {column} has been deleted.")
    elif process == 'simulate':
        log.info(f"Compressed data has successfully been stored in data column {outcol}.")
    else:
        log.info(f"Data has successfully been compressed but not stored in the measurement set.")
   
    total_data_volume = data_volume_calc(num_polarisations=ncorr, nrows=nrow, num_channels=nchan, num_data_cols=0)
    total_data_volume_with_data = data_volume_calc(num_polarisations=ncorr, nrows=nrow, num_channels=nchan, num_data_cols=num_datacols)
    data_size = total_data_volume_with_data - total_data_volume
    data_corr_size = data_size / ncorr
    compressed_data_size = np.sum(np.ceil(data_corr_size * 1 / np.mean(cfs)))
    
    log.info(f"The Measurement set has {num_datacols} data columns with a total volume of {data_size}. Each data column has a volume of {data_size / num_datacols}.")
    log.info(f"The compressed data stored in {outcol} has a total volume of {compressed_data_size}")

    results = ObjDict({
        "compressed_data": compressed_result,
        "cf": da.array(cfs),
        "data_size": data_size,
        "compressed_data_size": compressed_data_size,
    })

    return results
        


        
    
    