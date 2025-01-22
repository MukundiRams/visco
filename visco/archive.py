import dask.array as da
import xarray as xr
import zarr
import os
from daskms import xds_from_table,xds_from_ms,xds_to_table
import numpy as np
from typing import Union
# from bdsvd.msodify import add_column,delete_column
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

            



def archive_visdata(ms, correlation='XX,XY,YX,YY', fieldid=0, ddid=0, scan=1,
                column='DATA', outfilename='Compressed_Data', compressionrank=None,
                autocorrelation=False, decorrelation=None,
                antlist=None, flagvalue=0):
    
    """
    Compress the visibility data and store the decomposition components
    in Zarr file.
    """
    
     
    maintable = xds_from_table(ms,taql_where=f"FIELD_ID={fieldid} AND DATA_DESC_ID={ddid} AND SCAN_NUMBER={scan}")[0]
    spw_table = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0]
    num_corr = len(correlation)
    compressed_data = copy(maintable[column].data)
    
    vis_data,nbaselines = decompose(ms=ms, correlation=correlation, fieldid=fieldid,
                         ddid=ddid, scan=scan, column=column,
                       autocorrelation=autocorrelation,antlist=antlist, flagvalue=flagvalue)
    
    
    
    zarr_output_path = os.path.join(os.getcwd(), "Output", f"{outfilename}.zarr")

    root = zarr.open(zarr_output_path, mode='w')
   
    chunk_size = (1, compressed_data.shape[1], compressed_data.shape[2])
    root.create_dataset('ANTENNA1', data=maintable.ANTENNA1.values)
    root.create_dataset('ANTENNA2', data=maintable.ANTENNA2.values)
    root.create_dataset('TIME',data=maintable.TIME.values)
    root.create_dataset('UVW',data=maintable.UVW.values)
    root.create_dataset('EXPOSURE',data=maintable.EXPOSURE.values)
    root.create_dataset('INTERVAL',data=maintable.INTERVAL.values)
    root.create_dataset('TIME_CENTROID',data=maintable.TIME_CENTROID.values)
    root.create_dataset('SCAN_NUMBER',data=maintable.SCAN_NUMBER.values)
    root.create_dataset('FIELD_ID',data=maintable.FIELD_ID.values)
    
    root.attrs["shape"] = compressed_data.shape
    root.attrs["chunks"] = compressed_data.chunks
    # root.attrs["datatype"] = compressed_data.dtype
    # print(compressed_data.dtype)
    
    #storing subtables
    spw = root.create_group('SPECTRAL_WINDOW')
    spw.create_dataset('CHAN_WIDTH',data=spw_table.CHAN_WIDTH.values )
    spw.create_dataset('CHAN_FREQ',data=spw_table.CHAN_FREQ.values )
    spw.create_dataset('NUM_CHAN',data=spw_table.NUM_CHAN.values )
    spw.create_dataset('TOTAL_BANDWIDTH',data=spw_table.TOTAL_BANDWIDTH.values )
    spw.create_dataset('RESOLUTION',data=spw_table.RESOLUTION.values )
    spw.create_dataset('EFFECTIVE_BW',data=spw_table.EFFECTIVE_BW.values )
    spw.create_dataset('MEAS_FREQ_REF',data=spw_table.MEAS_FREQ_REF.values )
    spw.create_dataset('REF_FREQUENCY',data=spw_table.REF_FREQUENCY.values )
    spw.create_dataset('FLAG_ROW',data=spw_table.FLAG_ROW.values )
    
    decomp_group = root.create_group('DECOMPOSITIONS')

    if compressionrank is not None and decorrelation is not None:
        raise ValueError(f"Only compressionrank or decorrelation should be provided, not both.")
    
    if decorrelation is not None:
        pass
    
    elif compressionrank is not None:
    
        for bli in vis_data:
            ant1name = vis_data[bli]["ant1name"]
            ant2name = vis_data[bli]["ant2name"]
            baseline_key = f"{ant1name}-{ant2name}"
            
            
            baseline_group = decomp_group.create_group(baseline_key)
            
            for corr in correlation.split(','):
                U, singvals, WT = vis_data[bli][corr].data
                fullrank = vis_data[bli][corr].rank
                m, n = vis_data[bli][corr].shape
                baseline_filter = vis_data[bli]["baseline_filter"]
                ci = vis_data[bli][corr].ci
                
                
                corr_group = baseline_group.create_group(corr)
                corr_group.create_dataset('U', data=U[:,:compressionrank].compute())
                corr_group.create_dataset('S', data=singvals[:compressionrank].compute())
                corr_group.create_dataset('WT', data=WT[:compressionrank,:].compute())
                
                
                corr_group.attrs.update({
                    'baseline_filter': baseline_filter.tolist(),  
                    'ci': ci,
                    'shape': (m, n),
                    'fullrank': fullrank,
                
                })
                
                
        
        log.info(f"Data and decomposition components successfully stored at {zarr_output_path}")
        
    else:
        raise ValueError(f"Neither compressionrank or decorrelation is provided. Please provide one.")  


    