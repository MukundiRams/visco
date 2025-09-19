import unittest
from visco.compress_ms import compress_full_ms
import os



    
ms = 'data/sim-visco-kat7.ms'
zarrf = 'data/visco_test.zarr'

def run_compression():
    
    compress_full_ms(ms_path=ms, zarr_path=zarrf, consolidated=True,
                    chunk_size_row=5000,
                    overwrite=True,compressor='zstd',
                    level=3,nworkers=1, nthreads=1, memory_limit='2GB',
                    direct_to_workers=False, silence_logs=True,
                    correlation='XX,YY',correlation_optimized=False,
                    fieldid=0,ddid=0,scan=1,column='DATA',
                    outcolumn='COMPRESSED_DATA',
                    dashboard_addr=None, decorrelation=0.90
                    )

def validate_zarr_existence():
    
    assert os.path.exists(zarrf)
    
def inspect_zarr_contents():
    
    import xarray as xr
    zr = xr.open_zarr(zarrf, consolidated=True,group='MAIN')
    print(zr)
    assert 'FLAG' in zr
    assert 'UVW' in zr
    assert 'ANTENNA1' in zr
    assert 'ANTENNA2' in zr
    assert 'TIME' in zr
    assert 'SCAN_NUMBER' in zr
    assert 'WEIGHT' in zr
    
    
    zrspw = xr.open_zarr(zarrf, consolidated=True,group='SPECTRAL_WINDOW')
    print(zrspw)
    assert 'NUM_CHAN' in zrspw
    assert 'CHAN_FREQ' in zrspw
    assert 'CHAN_WIDTH' in zrspw
    assert 'EFFECTIVE_BW' in zrspw
    
if __name__ == "__main__":
    run_compression()
    validate_zarr_existence()
    inspect_zarr_contents()
