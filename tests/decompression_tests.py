from visco.decompress_ms import write_datasets_to_ms
import os


zarrf = 'data/visco_test.zarr'
msname = 'data/visco_decompressed.ms'

def run_decompression():

    write_datasets_to_ms(zarr_path=zarrf,
                         msname=msname,
                         column='COMPRESSED_DATA')

def validate_ms_existence():

    assert os.path.exists(msname)
    
def inspect_ms_contents():
    
    from daskms import xds_from_table
    
    ds = xds_from_table('data/visco_decompressed.ms')[0]
    print(ds)
    
    assert 'DATA' in ds
    assert 'FLAG' in ds
    assert 'UVW' in ds
    assert 'ANTENNA1' in ds
    assert 'ANTENNA2' in ds
    assert 'TIME' in ds
    assert 'SCAN_NUMBER' in ds
    assert 'WEIGHT' in ds
    assert 'SIGMA' in ds
    
    dsspw = xds_from_table('data/visco_decompressed.ms::SPECTRAL_WINDOW')[0]
    
    print(dsspw)
    assert 'NUM_CHAN' in dsspw
    assert 'CHAN_FREQ' in dsspw
    assert 'CHAN_WIDTH' in dsspw 
    assert 'EFFECTIVE_BW' in dsspw
    

if __name__ == "__main__":
    run_decompression()
    validate_ms_existence()
    inspect_ms_contents()      