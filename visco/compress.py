import dask.array as da
import xarray as xr
import zarr
import os
from daskms import xds_from_table,xds_from_ms,xds_to_table
import numpy as np
from typing import Union
from visco.msmodify import add_column,delete_column
from visco.msvolume import data_volume_calc,compression_factor
# from scabha.basetypes import File
from copy import deepcopy,copy
from visco.utilities import ObjDict
import visco
log = visco.get_logger(name="BDSVD")
# from tqdm import tqdm
import logging
logging.getLogger("daskms").setLevel(logging.ERROR)
from omegaconf import OmegaConf



        
def decompose(ms,correlation, fieldid,ddid,scan,
                column,autocorrelation,antlist,flagvalue):
    

    """
    Decompose the visibility data matrix for each baseline and 
    store into a dictionary.
    """
    CORR_TYPES = OmegaConf.load(f"{visco.PCKGDIR}/ms_corr_types.yaml").CORR_TYPES
    CORR_TYPES_REVERSE = OmegaConf.load(f"{visco.PCKGDIR}/ms_corr_types_reverse.yaml").CORR_TYPES
    
    #open the measurement set and its subtables
    maintable = xds_from_table(ms)[0]
    # spwtable = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0]
    fieldtable = xds_from_table(f"{ms}::FIELD")[0]
    antennatable = xds_from_table(f"{ms}::ANTENNA")[0]
    # ddtable = xds_from_table(f"{ms}::DATA_DESCRIPTION")[0]
    poltable = xds_from_table(f"{ms}::POLARIZATION")[0]
    
    if scan not in maintable.SCAN_NUMBER.data:
        raise ValueError(f"Invalid selected SCAN_NUMBER {scan}.\
                            Available SCAN_NUMBERs are {da.unique(maintable.SCAN_NUMBER.data)}")
    
    if ddid not in maintable.DATA_DESC_ID.data:
        raise ValueError(f"Invalid selected DATA_DESC_ID {ddid}.\
                            Available DATA_DESC_IDs are {da.unique(maintable.DATA_DESC_ID.data)}")
    
    field_names = fieldtable.NAME.data
    if fieldid < 0 or fieldid >= field_names.shape[0]:
        raise ValueError(f"Invalid selected FIELD_ID {fieldid}. There are\
                            {field_names.shape[0]} fields.")
    
    log.info(f"Decomposing visibilities for DATA_DESC_ID {ddid}. FIELD_ID {fieldid}, and SCAN_NUMBER {scan}.")
    
    ant1 = maintable.ANTENNA1.data.compute()
    ant2 = maintable.ANTENNA2.data.compute()
 
    if autocorrelation:
        baselines = np.unique(list(zip(ant1,ant2)),axis=0)
    else:
        baselines = np.unique([(a1, a2) for a1, a2 in zip(ant1,ant2) if a1 != a2],axis=0)
    nbaselines = baselines.shape[0]
    
    if antlist:
        baselines = antlist
        nbaselines = len(baselines)

     
    corr_types = poltable.CORR_TYPE.data.compute()
    corr_list = []
    for corr in corr_types:
        for cor in corr:
            corr_name = CORR_TYPES_REVERSE[int(cor)]
            corr_list.append(corr_name)
    log.info(f"The following correlations are available:{corr_list}")
    
    corr_list_user = []
    for corr in correlation.split(','):
        corr_ind = CORR_TYPES[str(corr)]
        corr_list_user.append(corr_ind)
    log.info(f"The user has selected the following correlations:{list(correlation.split(','))}")
    
    
    fidmsk = (maintable.FIELD_ID.data.compute() == fieldid)
    didmsk = (maintable.DATA_DESC_ID.data.compute() == ddid)
    scnmsk = (maintable.SCAN_NUMBER.data.compute() == scan)
    
    vis_dict = {}
    for bx,(antenna1,antenna2) in enumerate(baselines):
        
        vis_dict[bx] = {}
        
        ant1msk = (maintable.ANTENNA1.data.compute() == antenna1)
        ant2msk = (maintable.ANTENNA2.data.compute() == antenna2)
        baseline_filter = fidmsk & didmsk & scnmsk & ant1msk & ant2msk
        
        uvw = maintable.UVW.data[baseline_filter]
        baseline_length = da.sqrt(uvw[:,0]**2 + uvw[:,1]**2)
        vis_dict[bx]["length"] = baseline_length
        vis_dict[bx]["baseline_filter"] = baseline_filter
        
        ant1name = antennatable.NAME.values[antenna1]
        ant2name = antennatable.NAME.values[antenna2]
        vis_dict[bx]["ant1name"] = ant1name
        vis_dict[bx]["ant2name"] = ant2name
        
        for c in corr_list_user:
            ci = np.where(corr_types == c)[0][0]
            flag = maintable.FLAG.data[baseline_filter,:,ci]
            vis_data = copy(maintable[column].data[baseline_filter,:,ci])
            vis_data[flag] = flagvalue
            U,singvals,WT = da.linalg.svd(vis_data)
            fullrank = min(vis_data.shape[0],vis_data.shape[1])
        
            corr_type = CORR_TYPES_REVERSE[c]
            
            
            vis_dict[bx][corr_type] = ObjDict({
                "data": (U, singvals, WT),
                "rank": fullrank,
                "reduced_rank": fullrank,
                "shape": vis_data.shape,
                "ci": ci,
            })
            
            
            log.info(f"Decomposing visibility data for baseline {ant1name}-{ant2name}"
                     f" and correlation {corr_type}")
    
    return vis_dict,nbaselines

            
def reconstruction(U,singular_vals,WT,fullrank,compressionrank=-1,type='cum'):
    """
    Reconstruct the truncated data to simulate lossy compression.
    """
    
    #pass operation if the compression rank<0 or >fullrank.
    if compressionrank <= 0 or compressionrank > fullrank:
        pass
    else:
        #if compression is using cumulative singular values.
        if type=='cum':
            truncated_singular_vals = copy(singular_vals)
            truncated_singular_vals[compressionrank:] = 0.0
            
            compressed_data = U @ da.diag(truncated_singular_vals) @ WT
        elif type=='ind':
            # truncate the unwanted singular values
            truncated_singular_vals = singular_vals[compressionrank -1] #starting index is 0.
            truncated_U = U[:,compressionrank].reshape(-1,1) 
            truncated_WT = WT[compressionrank ,:].reshape(1,-1)

            compressed_data = truncated_U @ (truncated_singular_vals * truncated_WT)
            
        return compressed_data


def compress_visdata(ms, correlation='XX,XY,YX,YY', fieldid=0, ddid=0, scan=1,
                column='DATA', outcol='COMPRESSED_DATA', compressionrank=None,type='cum',
                decorrelation=None, process='simulate', autocorrelation=False,
                antlist=None, flagvalue=0, zarr_output_path=None):
    
    """
    Compress the visibility data.
    """
    maintable = xds_from_table(ms)[0]
    num_corr = len(correlation)
    compressed_data = copy(maintable[column].data)
    
    
    vis_data,nbaselines = decompose(ms=ms, correlation=correlation, fieldid=fieldid,
                         ddid=ddid, scan=scan, column=column,
                       autocorrelation=autocorrelation,antlist=antlist, flagvalue=flagvalue)
    
    
    
    if compressionrank is not None and decorrelation is not None:
        raise ValueError(f"Only compressionrank or decorrelation should be provided, not both.")

    if decorrelation is not None:
        pass
    
    elif compressionrank is not None:
        total_data_size = 0
        total_compressed_size = 0
        
        # decomp_components = {
        #     'U': {},
        #     'singvals': {},
        #     'WT': {},
        #     'baseline_info': {}
        # }
        
        recon_dict = {}
        for bli in vis_data:
            recon_dict[bli] = {}
            for corr in correlation.split(','):
                U,singvals,WT = vis_data[bli][corr].data
                fullrank = vis_data[bli][corr].rank
                m,n = vis_data[bli][corr].shape
                baseline_filter = vis_data[bli]["baseline_filter"]
                ci = vis_data[bli][corr].ci
                ant1name = vis_data[bli]["ant1name"]
                ant2name = vis_data[bli]["ant2name"]
                
                compressed_visdata = reconstruction(U,singvals,WT,fullrank,compressionrank,type)
                compressed_data[baseline_filter, :, ci] = compressed_visdata
                # recon_dict[bli][corr] = {
                #     "compressed_data": compressed_visdata,
                # }
                # maintable[outcol].data[baseline_filter,:,ci]  = compressed_visdata
                
                compressionratio = compression_factor(m,n,compressionrank)

                log.info(f"The visibility data for baseline {ant1name}-{ant2name} and"
                         f" correlation {corr} has been compressed and stored in"
                         f" {outcol} with a compression factor of {compressionratio}.")
                
            
                
                baseline_data_volume = data_volume_calc(num_polarisations=num_corr,nrows=m,num_channels=n,
                                                     auto_correlations=autocorrelation,num_data_cols=0)
                baseline_data_volume_with_data = data_volume_calc(num_polarisations=1,nrows=m,num_channels=n,
                                                     auto_correlations=autocorrelation,num_data_cols=1)
                baseline_data_size = baseline_data_volume_with_data - baseline_data_volume
                baseline_compressed_size = np.ceil(baseline_data_size/compressionratio)
                
                total_data_size += baseline_data_size
                total_compressed_size += baseline_compressed_size
                
                
        
        add_column(ms,outcol,compressed_data)
        if process=='simulate':
            pass
        elif process == 'archive':
            delete_column(ms,column)
        else:
            raise ValueError(f"Process can only be 'simulate' or 'archive'. Please choose one.")
    
        
        
        log.info(f"Visibility data has been compressed"
                 f" from data size of {total_data_size/  1e9:.2e} GB to a compressed"
                 f" data size of {total_compressed_size/  1e9:.2e} GB")
    
    
    else:
        raise ValueError(f"Neither compressionrank or decorrelation is provided. Please provide one.")
    


