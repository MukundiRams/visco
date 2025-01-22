import dask.array as da
from copy import copy


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