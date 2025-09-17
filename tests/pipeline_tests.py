import unittest
from visco.compress_ms import compress_full_ms
from visco.decompress_ms import write_datasets_to_ms
import os


class TestArchiveFunction(unittest.TestCase):
    
    def setUp(self):
        self.test_ms = "sim-visco-kat7.ms"
        self.compressed_file = "compressed-data.zarr"
    
    def test_compression_decompression(self):
        """Test the full archival and decompression pipeline."""
        
        # Step 1: Compress the data
        compress_full_ms(
            ms_path=self.test_ms,
            zarr_path=f"{self.compressed_file}",
            consolidated=True,
            chunk_size_row=5000,
            overwrite=True,
            compressor='zstd',
            level=4,
            nworkers=1,
            nthreads=1,
            memory_limit="2GB",
            direct_to_workers=False,
            fieldid=0,
            ddid=0,
            scan=1,
            correlation = 'XX,YY',
            correlation_optimized=False,
            column='DATA',
            outcolumn='COMPRESSED_DATA',
            compressionrank=1,
            
        )
        self.assertTrue(os.path.exists(f"{self.compressed_file}"))

        # Step 2: Decompress the data
        write_datasets_to_ms(
            zarr_path=f"{self.compressed_file}",
            column='COMPRESSED_DATA',
            msname="decompressed.ms"
            
        )
        self.assertTrue(os.path.exists("decompressed.ms"))

if __name__ == "__main__":
    unittest.main()
