import dask.array as da
import xarray as xr
import zarr
import os
import ast
from itertools import combinations
from daskms import xds_from_table,xds_from_ms,xds_to_table
import numpy as np
from typing import List
# from visco.msvolume import data_volume_calc,compression_factor
# from scabha.basetypes import File
from copy import copy
from tqdm.dask import TqdmCallback
from visco.utilities import ObjDict
import visco
log = visco.get_logger(name="VISCO")
# from tqdm import tqdm
import logging
logging.getLogger("daskms").setLevel(logging.ERROR)

from omegaconf import OmegaConf


def openms(ms,sub=None,query=None):
    
    if sub:
        if query:
            ds = xds_from_table(f"{ms}::{sub}",taql_where=query)
        else:
            ds = xds_from_table(f"{ms}::{sub}")
    else:
        if query:
            ds = xds_from_table(ms,taql_where=query)
        else:
            ds = xds_from_table(ms)
        
    for dsi in ds:
        dataset = dsi
    return dataset
        
      
def decompose(ms,correlation, fieldid,ddid,scan,
                column,autocorrelation,antlist,flagvalue,maintab,fieldtab,anttab,poltab):
    

    """
    Decompose the visibility data matrix for each baseline and 
    store into a dictionary.
    """
    CORR_TYPES = OmegaConf.load(f"{visco.PCKGDIR}/ms_corr_types.yaml").CORR_TYPES
    CORR_TYPES_REVERSE = OmegaConf.load(f"{visco.PCKGDIR}/ms_corr_types_reverse.yaml").CORR_TYPES
    
    #open the measurement set and its subtables
    
    maintable = maintab
    fieldtable = fieldtab
    antennatable = anttab
    poltable = poltab
    
    scan_number = maintable.SCAN_NUMBER.data
    data_desc_id = maintable.DATA_DESC_ID.data
    field_id = maintable.FIELD_ID.data
    ant1 = maintable.ANTENNA1.data.compute()
    ant2 = maintable.ANTENNA2.data.compute()
    corr_types = poltable.CORR_TYPE.data.compute()
    field_names = fieldtable.NAME.data
    
    fidmsk = (field_id.compute() == fieldid)
    didmsk = (data_desc_id.compute() == ddid)
    scnmsk = (scan_number.compute() == scan)
    
    if scan not in scan_number:
        available_scans = da.unique(scan_number).compute().tolist()
        raise ValueError(f"Invalid SCAN_NUMBER {scan}. Available scans are: {available_scans}")
        # raise ValueError(f"Invalid selected SCAN_NUMBER {scan}.\
        #                     Available SCAN_NUMBERs are {da.unique(maintable.SCAN_NUMBER.data)}")
    
    if ddid not in data_desc_id:
        raise ValueError(f"Invalid selected DATA_DESC_ID {ddid}.\
                            Available DATA_DESC_IDs are {da.unique(data_desc_id)}")
    
    
    if fieldid < 0 or fieldid >= field_names.shape[0]:
        raise ValueError(f"Invalid selected FIELD_ID {fieldid}. There are\
                            {field_names.shape[0]} fields.")
    
    log.info(f"Decomposing visibilities for DATA_DESC_ID {ddid}. FIELD_ID {fieldid}, and SCAN_NUMBER {scan}.")
    
 
    
    if antlist:
        
        if isinstance(antlist, str): 
            try:
                antlist = ast.literal_eval(antlist)
                if not isinstance(antlist, list) or not all(isinstance(x, int) for x in antlist):
                    raise ValueError("Parsed antlist is not a valid list of integers.")
            except (ValueError, SyntaxError):
                raise ValueError(f"Invalid format for antlist: {antlist}. Expected a list of integers.")
        
        baselines = list(combinations(antlist, 2))
        nbaselines = len(baselines)
        
        
    else:
        if autocorrelation:
            baselines = np.unique(list(zip(ant1,ant2)),axis=0)
        else:
            baselines = np.unique([(a1, a2) for a1, a2 in zip(ant1,ant2) if a1 != a2],axis=0)
        nbaselines = baselines.shape[0]
        

     
    
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
    
    
    
    
    vis_dict = {}
    for bx,(antenna1,antenna2) in enumerate(baselines):
        
        vis_dict[bx] = {}
        
        ant1msk = (ant1 == antenna1)
        ant2msk = (ant2 == antenna2)
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
                "corr": corr_type,
            })
            
            
            log.info(f"Decomposing visibility data for baseline {ant1name}-{ant2name}"
                     f" and correlation {corr_type}")
    
    return vis_dict

            



def archive_visdata(ms, correlation='XX,XY,YX,YY', fieldid=0, ddid=0, scan=1,
                column='DATA', outfilename='compressed-data.zarr', compressionrank=None,
                autocorrelation=False, decorrelation=None,
                antlist=None, flagvalue=0):
    
    """
    Compress the visibility data and store the decomposition components
    in Zarr file.
    """
    
    if compressionrank is not None and decorrelation is not None:
        raise ValueError("Only one of 'compressionrank' or 'decorrelation' should be provided, not both.")
    elif compressionrank is None and decorrelation is None:
        raise ValueError("Either 'compressionrank' or 'decorrelation' must be provided.")
    
    maintable = openms(ms,query = f"FIELD_ID={fieldid} AND DATA_DESC_ID={ddid} AND SCAN_NUMBER={scan}")
    mtab = openms(ms)
    spw_table = openms(ms,"SPECTRAL_WINDOW")
    ant_table = openms(ms,"ANTENNA")
    pol_table = openms(ms,"POLARIZATION")
    fld_table = openms(ms,"FIELD")
    pntng_table = openms(ms,"POINTING")
    
    
    num_corr = len(correlation)
    compressed_data = copy(maintable[column].data)
    
    vis_data = decompose(ms=ms, correlation=correlation, fieldid=fieldid,
                         ddid=ddid, scan=scan, column=column,
                       autocorrelation=autocorrelation,antlist=antlist, flagvalue=flagvalue, 
                       maintab=mtab,fieldtab=fld_table,anttab=ant_table,poltab=pol_table)
    
    
    
    zarr_output_path = os.path.join(os.getcwd(), "Output", f"{outfilename}")

    root = zarr.open(zarr_output_path, mode='w')
   
   #storing main table
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
    
    #storing spectral window table
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

    
     #storing polarization table
    pol = root.create_group('POLARIZATION')
    pol.create_dataset('CORR_TYPE',data = pol_table.CORR_TYPE.values)
    pol.create_dataset('CORR_PRODUCT',data = pol_table.CORR_PRODUCT.values)
    pol.create_dataset('NUM_CORR',data = pol_table.NUM_CORR.values)
    
     #storing field table
    fld = root.create_group('FIELD')
    fld.create_dataset('PHASE_DIR', data=fld_table.PHASE_DIR.values)
    fld.create_dataset('DELAY_DIR', data=fld_table.DELAY_DIR.values)
    fld.create_dataset('REFERENCE_DIR', data=fld_table.REFERENCE_DIR.values)
    
     #storing pointing table
    pntng = root.create_group('POINTING')
    pntng.create_dataset('TARGET', data=pntng_table.TARGET.values)
    pntng.create_dataset('TRACKING', data=pntng_table.TRACKING.values)
    pntng.create_dataset('DIRECTION', data=pntng_table.DIRECTION.values)
    pntng.create_dataset('TIME', data=pntng_table.TIME.values)
    pntng.create_dataset('INTERVAL', data=pntng_table.INTERVAL.values)
    
    #storing antenna table
    ant = root.create_group('ANTENNA')
    # ant.create_dataset('NAME',data=ant_table.NAME.values)
    ant.create_dataset('DISH_DIAMETER',data=ant_table.DISH_DIAMETER.values)
    # ant.create_dataset('MOUNT',data=ant_table.MOUNT.values)
    ant.create_dataset('POSITION',data=ant_table.POSITION.values)
    # ant.create_dataset('TYPE',data=ant_table.TYPE.values)
    
    decomp_group = root.create_group('DECOMPOSITIONS')

    
    if decorrelation is not None:
        #Do this later
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
                # ci = vis_data[bli][corr].ci
                
                
                
                corr_group = baseline_group.create_group(corr)
                corr_group.create_dataset('U', data=U[:,:compressionrank].compute())
                corr_group.create_dataset('S', data=singvals[:compressionrank].compute())
                corr_group.create_dataset('WT', data=WT[:compressionrank,:].compute())
                
                
                corr_group.attrs.update({
                    'baseline_filter': baseline_filter.tolist(),  
                    'corr': corr,
                    'shape': (m, n),
                    'fullrank': fullrank,
                
                })
                
                
        
        log.info(f"Data successfully stored at {zarr_output_path}")
     

# archive_visdata('/home/mukundi/averaging/stimela_scripts/msdir/sim-svd-0.66deg.ms',correlation='XX',compressionrank=10,antlist=[1,2,3])

    