import dask.array as da
import xarray as xr
import os
import ast
from itertools import combinations
from daskms import xds_from_table
import numpy as np
from copy import copy
from tqdm.dask import TqdmCallback
from visco.utilities import ObjDict
import visco
log = visco.get_logger(name="VISCO")
import logging
logging.getLogger("daskms").setLevel(logging.ERROR)

from omegaconf import OmegaConf


def openms(ms,sub=None,query=None):
    """
        Open a Meaurement Set.
        
        Params
        ----
        ms: str 
        - The path to the MS.
        
        sub: str 
        - The name of the subtable to be opened.
        
        query: str 
        - A query to send to the MS.
        
        Returns
        ----
        
        An xarray dataset.
        
    """
    #If subtable is given
    if sub:
        #If query is given
        if query:
            ds = xds_from_table(f"{ms}::{sub}",taql_where=query)
        else:
            ds = xds_from_table(f"{ms}::{sub}")
    else:
        #If opening the main table
        if query:
            ds = xds_from_table(ms,taql_where=query)
        else:
            ds = xds_from_table(ms)
     
    #Accessing all the chunks in the dataset   
    for dsi in ds:
        dataset = dsi
    return dataset
        
      
def decompose(ms,correlation, fieldid,ddid,scan,
                column,autocorrelation,antlist,flagvalue,maintab,fieldtab,anttab,poltab):
    

    """
    Decompose the visibility data matrix for each baseline and 
    store into a dictionary.
    
    Params
    ----
    
    ms: str
    - The path to the MS.
    
    correlation: str
    - The correlation to decompose.
    
    fieldid: int
    - The FIELD_ID to use for the decomposition.
    
    ddid: int
    - The DATA_DESC_ID to use for the decomposition.
    
    scan: int
    - The SCAN_NUMBER to use.
    
    column: str
    - The data column to decompose.
    
    autocorrelation: bool
    - Whether there is autocorrelation or not.
    
    antlist: List[int]
    - The antenna indices to use.
    
    flagvalue: int or float
    - The value to replace the data that is flagged.
    
    maintab: xarray dataset
    - The main table of the measurement set.
    
    fieldtab: xarray dataset
    - The FIELD table of the measurement set.
    
    anttab: xarray dataset
    - The ANTENNA table of the measurement set.
    
    poltab: xarray dataset
    - The POLARIZATION table of the measurement set.
    
    Returns 
    ---
    visdata_dict: dict
    - A dictionary with the decomposed data.
    
    """
    CORR_TYPES = OmegaConf.load(f"{visco.PCKGDIR}/ms_corr_types.yaml").CORR_TYPES
    CORR_TYPES_REVERSE = OmegaConf.load(f"{visco.PCKGDIR}/ms_corr_types_reverse.yaml").CORR_TYPES
    
    #Open the measurement set and its subtables
    maintable = maintab
    fieldtable = fieldtab
    antennatable = anttab
    poltable = poltab
    
    #Get the needed data
    scan_number = maintable.SCAN_NUMBER.data
    data_desc_id = maintable.DATA_DESC_ID.data
    field_id = maintable.FIELD_ID.data
    ant1 = maintable.ANTENNA1.data.compute()
    ant2 = maintable.ANTENNA2.data.compute()
    corr_types = poltable.CORR_TYPE.data.compute()
    field_names = fieldtable.NAME.data
    
    #Create a mask 
    fidmsk = (field_id.compute() == fieldid)
    didmsk = (data_desc_id.compute() == ddid)
    scnmsk = (scan_number.compute() == scan)
    
    #Check if the given scan number, ddid, and fieldid exists.
    if scan not in scan_number:
        available_scans = da.unique(scan_number).compute().tolist()
        raise ValueError(f"Invalid SCAN_NUMBER {scan}. Available scans are: {available_scans}")
        
    
    if ddid not in data_desc_id:
        raise ValueError(f"Invalid selected DATA_DESC_ID {ddid}.\
                            Available DATA_DESC_IDs are {da.unique(data_desc_id)}")
    
    
    if fieldid < 0 or fieldid >= field_names.shape[0]:
        raise ValueError(f"Invalid selected FIELD_ID {fieldid}. There are\
                            {field_names.shape[0]} fields.")
    
    log.info(f"Decomposing visibilities for DATA_DESC_ID {ddid}. FIELD_ID {fieldid}, and SCAN_NUMBER {scan}.")
     
    #If given the antenna indices, use them to get baselines. 
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
        #If there are autocorrelations, which probably will never happen, consider it.
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
    
    #Initialize the data dictionary to contain the decomposed components.
    vis_dict = {}
    
    #Go through all the baselines.
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
        
        #Go through the given correlations
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
    
    Parameters
    ----
    
    ms: str
    - The path to the MS.
    
    correlation: str
    - The correlation to decompose.
    - default is 'XX,XY,YX,YY'.
    
    fieldid: int
    - The FIELD_ID to use for the decomposition.
    - Default is 0.
    
    ddid: int
    - The DATA_DESC_ID to use for the decomposition.
    - Default is 0.
    
    scan: int
    - The SCAN_NUMBER to use.
    - Default is 1.
    
    column: str
    - The data column to decompose.
    - Default is DATA.
    
    autocorrelation: bool
    - Whether there is autocorrelation or not.
    - Default is False.
    
    antlist: List[int]
    - The antenna indices to use.
    
    flagvalue: int or float
    - The value to replace the data that is flagged.
    - Default is 0.
    
    outfilename: str
    - The name of the output zarr file.
    - Default is compressed-data.zarr.
    
    compressionrank: int
    - The compression rank to apply uniformly on all the baselines.
    
    decorrelation: float
    - The minimum signal loss percentage on each baseline.
    
    Returns
    ----
    A zarr file with the data.
    """
    
    # Only one should be provided. Cannot provide both decorrelation and compressionrank
    if compressionrank is not None and decorrelation is not None:
        raise ValueError("Only one of 'compressionrank' or 'decorrelation' should be provided, not both.")
    elif compressionrank is None and decorrelation is None:
        raise ValueError("Either 'compressionrank' or 'decorrelation' must be provided.")
    
    #Open the tables
    maintable = openms(ms,query = f"FIELD_ID={fieldid} AND DATA_DESC_ID={ddid} AND SCAN_NUMBER={scan}")
    mtab = openms(ms)
    spw_table = openms(ms,"SPECTRAL_WINDOW")
    ant_table = openms(ms,"ANTENNA")
    pol_table = openms(ms,"POLARIZATION")
    fld_table = openms(ms,"FIELD")
    pntng_table = openms(ms,"POINTING")
    pointing_chunks = pntng_table.chunks["row"][0]
    data_row_chunks = maintable.chunks["row"][0]
    
    compressed_data = copy(maintable[column].data)
    
    #Full rank of the data.
    fullrank = min(compressed_data.shape[0],compressed_data.shape[1])
    log.info(f"The full rank of the data is {fullrank}.")
    
    #Get the decompositions.
    vis_data = decompose(ms=ms, correlation=correlation, fieldid=fieldid,
                         ddid=ddid, scan=scan, column=column,
                       autocorrelation=autocorrelation,antlist=antlist, flagvalue=flagvalue, 
                       maintab=mtab,fieldtab=fld_table,anttab=ant_table,poltab=pol_table)
    
    
    #The path to the Zarr file containing the compression results
    zarr_output_path = os.path.join(os.getcwd(), "zarr-output", f"{outfilename}")
    
    #Store the main table in the zarr file.
    ds = xr.Dataset(
    {
        "ANTENNA1": (("row"), maintable.ANTENNA1.values),
        "ANTENNA2": (("row"), maintable.ANTENNA2.values),
        "TIME": (("row"), maintable.TIME.values),
        "UVW": (("row", "uvw_dim"), maintable.UVW.values),
        "EXPOSURE": (("row"), maintable.EXPOSURE.values),
        "INTERVAL": (("row"), maintable.INTERVAL.values),
        "TIME_CENTROID": (("row"), maintable.TIME_CENTROID.values),
        "SCAN_NUMBER": (("row"), maintable.SCAN_NUMBER.values),
        "FIELD_ID": (("row"), maintable.FIELD_ID.values),
    },
    coords={
        "row": np.arange(len(maintable.ANTENNA1)),  
    },
    attrs={
        "shape": compressed_data.shape,
        "chunks": compressed_data.chunks,
        "ROWID": maintable.ROWID.values,
        },
    )
    
    #Writing to the zarr file.
    ds = ds.chunk({"row": data_row_chunks})
    ds.to_zarr(zarr_output_path, mode="w")
    
    #Compress the flags using numpy packbits. Groups 8 booleans into one interger.
    orig_flags_row = maintable.FLAG_ROW.values
    orig_flags = maintable.FLAG.values
    packed_flags_row = compress_bits(orig_flags_row,dim=1)
    packed_flags = compress_bits(orig_flags,dim=3)
    
    #Write the flag data to the zarr file.
    ds_flag_row = xr.Dataset({
        "FLAG_ROW":(("row"), packed_flags_row)
        },
        coords={
            "row": np.arange(packed_flags_row.shape[0])
        }
    )
    
    ds_flag_row.to_zarr(zarr_output_path, group="FLAG_ROW", mode="w")
    
    ds_flag = xr.Dataset({
        "FLAG":(("row"), packed_flags)
        },
        coords={
            "row": np.arange(packed_flags.shape[0])
        }
    )
    
    ds_flag.to_zarr(zarr_output_path, group="FLAG", mode="w")

    ds_spw = xr.Dataset(
        {
            "CHAN_WIDTH":(("row","chan"),spw_table.CHAN_WIDTH.values),
            "CHAN_FREQ":(("row","chan"),spw_table.CHAN_FREQ.values),
            "EFFECTIVE_BW":(("row","chan"),spw_table.EFFECTIVE_BW.values),
            "RESOLUTION":(("row","chan"),spw_table.RESOLUTION.values),
            "NUM_CHAN":(("row"),spw_table.NUM_CHAN.values),
            "REF_FREQUENCY":(("row"),spw_table.REF_FREQUENCY.values),
            "MEAS_FREQ_REF":(("row"),spw_table.MEAS_FREQ_REF.values),
            "TOTAL_BANDWIDTH":(("row"),spw_table.TOTAL_BANDWIDTH.values),
            "FLAG_ROW":(("row"),spw_table.FLAG_ROW.values)
        },
        coords={
            "row": np.arange((spw_table.CHAN_WIDTH.shape[0])),
            "chan": np.arange((spw_table.CHAN_WIDTH.shape[1]))
        }
    )
    
    ds_spw.to_zarr(zarr_output_path, group="SPECTRAL_WINDOW", mode="w")
    
    
    ds_pol = xr.Dataset(
        {
            "CORR_TYPE":(("row","corr"),pol_table.CORR_TYPE.values),
            "CORR_PRODUCT":(("row","corr","corrprod_idx"),pol_table.CORR_PRODUCT.values),
            "NUM_CORR":(("row"),pol_table.NUM_CORR.values)
        },
        coords={
            "row": np.arange((pol_table.CORR_TYPE.shape[0])),
            "corr": np.arange((pol_table.CORR_TYPE.shape[1])),
            "corrprod_idx": np.arange((pol_table.CORR_PRODUCT.shape[2]))
        }
    )
    
    ds_pol.to_zarr(zarr_output_path, group="POLARIZATION", mode="w") 
    
    ds_fld = xr.Dataset(
        {
           "PHASE_DIR":(("row", "field-poly", "field-dir"),fld_table.PHASE_DIR.values),
           "DELAY_DIR":(("row", "field-poly", "field-dir"),fld_table.DELAY_DIR.values),
           "REFERENCE_DIR":(("row", "field-poly", "field-dir"),fld_table.REFERENCE_DIR.values)
        },
        coords={
            "row": np.arange((fld_table.PHASE_DIR.shape[0])),
            "field-poly": np.arange((fld_table.PHASE_DIR.shape[1])),
            "field-dir": np.arange((fld_table.PHASE_DIR.shape[2]))
        }
    )
    
    ds_fld.to_zarr(zarr_output_path, group="FIELD", mode="w")
    
    
    ds_pointing = xr.Dataset(
        {
          "TARGET":(("row", "point-poly", "radec"),pntng_table.TARGET.values),
          "TIME":(("row"),pntng_table.TIME.values),
          "INTERVAL":(("row"),pntng_table.INTERVAL.values),
          "TRACKING":(("row"),pntng_table.TRACKING.values),
          "DIRECTION":(("row", "point-poly", "radec"),pntng_table.DIRECTION.values)
        },
        coords={
            "row": np.arange((pntng_table.TARGET.shape[0])),
            "point-poly": np.arange((pntng_table.TARGET.shape[1])),
            "radec": np.arange((pntng_table.TARGET.shape[2]))
        }
    )
    
    ds_pointing = ds_pointing.chunk({"row": pointing_chunks})
    
    ds_pointing.to_zarr(zarr_output_path, group="POINTING", mode="w")

    
    ds_ant = xr.Dataset(
        {
           "DISH_DIAMETER":(("row"),ant_table.DISH_DIAMETER.values),
           "POSITION":(("row","pos_dim"),ant_table.POSITION.values),
           "MOUNT":(("row"),ant_table.MOUNT.values),
           "TYPE":(("row"),ant_table.TYPE.values),
            "NAME":(("row"),ant_table.NAME.values) 
        },
        coords={
            "row": np.arange((ant_table.DISH_DIAMETER.shape[0])),
            "pos_dim": np.arange((ant_table.POSITION.shape[1]))
        }
    )
    
    ds_ant.to_zarr(zarr_output_path, group="ANTENNA", mode="w")
    
    #If decorrelation is provided, compute the required singular values for each baselines
    #based on the desired signal loss.
    if decorrelation is not None:
        
        #Go through all the baselines and correlations
        for bli in vis_data:
            ant1name = vis_data[bli]["ant1name"]
            ant2name = vis_data[bli]["ant2name"]
            baseline_key = f"{ant1name}-{ant2name}"
            
            for corr in correlation.split(','):
                
                U, singvals, WT = vis_data[bli][corr].data
                sum_total = da.sum(singvals**2).compute()
                threshold = (decorrelation)**2 * sum_total
                cumulative = da.cumsum(singvals**2).compute()
                
                n = np.argmax(cumulative >= threshold) + 1  #+1 to convert index to count
                if n == 0:
                    n = len(singvals)

                print(f"Baseline: {baseline_key}, n: {n}")
                m, n_orig = vis_data[bli][corr].shape
                baseline_filter = vis_data[bli]["baseline_filter"]
   
                ds_decomp = xr.Dataset(
                {
                "U": (("time", "mode"), U[:, :n].compute()),  
                "S": (("mode",), singvals[:n].compute()),
                "WT": (("mode", "channel"), WT[:n, :].compute()),
                },
                coords={
                "time": np.arange(U.shape[0]),
                "mode": np.arange(singvals.shape[0])[:n],
                "channel": np.arange(WT.shape[1]),
                },
                attrs={
                "baseline_filter": baseline_filter.tolist(),
                },
                )
                
                group_path = f"DATA/{baseline_key}/{corr}"
                ds_decomp.to_zarr(zarr_output_path, group=group_path, mode="w")
                
    
    #If compressionrank is given, use it uniformly across all the baselines.
    elif compressionrank is not None:
    
        for bli in vis_data:
            ant1name = vis_data[bli]["ant1name"]
            ant2name = vis_data[bli]["ant2name"]
            baseline_key = f"{ant1name}-{ant2name}"
             
            for corr in correlation.split(','):
                U, singvals, WT = vis_data[bli][corr].data
                m, n = vis_data[bli][corr].shape
                baseline_filter = vis_data[bli]["baseline_filter"]
                
                ds_decomp = xr.Dataset(
                {
                "U": (("time", "mode"), U[:, :compressionrank].compute()), 
                "S": (("mode",), singvals[:compressionrank].compute()),
                "WT": (("mode", "channel"), WT[:compressionrank, :].compute()),
                },
                coords={
                "time": np.arange(U.shape[0]),
                "mode": np.arange(singvals.shape[0])[:compressionrank],
                "channel": np.arange(WT.shape[1]),
                },
                attrs={
                "baseline_filter": baseline_filter.tolist(),
                "corr": corr,
                },
                )
                
                group_path = f"DATA/{baseline_key}/{corr}"
                
                ds_decomp.to_zarr(zarr_output_path, group=group_path, mode="w")
    
        log.info(f"Data successfully stored at {zarr_output_path}")
     

def compress_bits(data_array,axis=None,dim=1):
    """"
    Compress data (flags) using numpy packbits.
    
    Params
    ----
    data_array: array
    - The data to be compressed.
    
    axis: int
    - The axis along which to compress the data.
    
    dim: int
    - The dimensions of the data.
    
    Returns
    ----
    
    The compressed data that is 8 times smaller in size than the original.
    """
    #If the data has one dimension, its easy.
    if dim==1:
        #If the data is not divisible by 8, padd it at the end with booleans so that it is divisible.
        if axis==None and data_array.shape[0] % 8 != 0:
            total_length = data_array.shape[0]
            pad_size = (8 - total_length % 8) % 8
            padded_data = np.pad(data_array, ((0, pad_size)), mode='constant', constant_values=False)
            packed_data = np.packbits(padded_data,axis=None)
            
        #if the data is divisible by 8, then its easy.
        elif axis==None and data_array.shape[0] % 8 == 0:
            packed_data = np.packbits(data_array,axis=None)
            
    #If the data has more than one dimension, flatten the array.
    elif dim>1:
        
        # if the flattened array is not divisibble by 8, then pad its end.
        if axis==None and data_array.size % 8 != 0:
            total_length = data_array.size
            pad_size = (8 - total_length % 8) % 8
            padded_data = np.pad(data_array, ((0, pad_size)), mode='constant', constant_values=False)
            packed_data = np.packbits(padded_data,axis=None)

        #if the flattenen array is divisible by 8, its easy.
        elif axis==None and data_array.size % 8 ==0:
            packed_data = np.packbits(data_array,axis=None)
        
    return packed_data

    
def compress_quantized(weight_array):
     
    quantized = weight_array.astype(np.float16)
    
    return quantized