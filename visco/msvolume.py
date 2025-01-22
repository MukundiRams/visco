import functools

def data_volume_calc(num_polarisations,
                     nrows,
                     num_channels, 
                     auto_correlations=False,
                     num_data_cols=1,
                     bits_per_vis = 64.0,
                     bits_exposure = 64.0,  # double precision
                     bits_per_flag = 8.0,
                     bits_per_weight = 32.0,
                     indexer_size = 32.0,
                     time_size = 64.0,  # double precision
                     flag_categories = 1,
                     uvw_size = 3 * 64.0  # double precision
                     ):
    """Calculates the size of an MSv2, considering data columns, flags
    (including flag category) and weights (including weight spectrum)
    Returns in units of bytes

    Adapted from SARAO calculator backend by the original author
    """

    # CASA memo 229 (Measurement Set v2 specification)
    # See https://casa.nrao.edu/Memos/229.html
    # Bits used per column (including non-compulsory
    # columns currently being dumped in
    # https://github.com/ska-sa/katdal/blob/master/katdal/ms_extra.py)
    # these include
    BITS_PER_ROW = {
        # Foreign keys
        "ANTENNA1": indexer_size,
        "ANTENNA2": indexer_size,
        "ARRAY_ID": indexer_size,
        "DATA_DESC_ID": indexer_size,
        "FEED1": indexer_size,
        "FEED2": indexer_size,
        "FIELD_ID": indexer_size,
        "OBSERVATION_ID": indexer_size,
        "PROCESSOR_ID": indexer_size,
        "SCAN_NUMBER": indexer_size,
        "STATE_ID": indexer_size,
        # complex valued data
        "DATA": bits_per_vis * num_polarisations * num_channels,
        # meta data
        "EXPOSURE": bits_exposure,
        "FLAG": bits_per_flag * num_polarisations * num_channels,
        "FLAG_CATEGORY": bits_per_flag
        * flag_categories
        * num_polarisations
        * num_channels,
        "FLAG_ROW": bits_per_flag,
        "WEIGHT_SPECTRUM": bits_per_weight * num_polarisations * num_channels,
        "SIGMA_SPECTRUM": bits_per_weight * num_polarisations * num_channels,
        "INTERVAL": bits_exposure,
        "SIGMA": bits_per_weight * num_polarisations,
        "WEIGHT": bits_per_weight * num_polarisations,
        "TIME": time_size,
        "TIME_CENTROID": time_size,
        "UVW": uvw_size,
    }
    # DATA is already added
    # add flexibility to add for instance CORRECTED_DATA and MODEL_DATA
    # commonly needed in reductions
    if num_data_cols == 0:
        del BITS_PER_ROW["DATA"]
    else:
        for i in range(num_data_cols - 1):
            BITS_PER_ROW[f"EXTRACOL{i}"] = BITS_PER_ROW["DATA"]

    # Calculate the number of bits per column of data, flags, and weights
    total_bits = functools.reduce(lambda a, b: a + b, BITS_PER_ROW.values())

    total_data_volume = nrows * total_bits / 8

    return total_data_volume


def compression_factor(m,n,compressionrank):
    
    originalsize = m*n
    newsize = 1*(m+n+0.5)
    
    cf = originalsize/newsize
    
    return cf
    