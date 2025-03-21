import unittest
from visco.archive import archive_visdata
from visco.decompress import decompress_visdata
import os


class TestArchiveFunction(unittest.TestCase):
    
    def setUp(self):
        self.test_ms = "sim-visco-kat7.ms"
        self.compressed_file = "compressed-data.zarr"
    
    def test_compression_decompression(self):
        """Test the full archival and decompression pipeline."""
        
        # Step 1: Compress the data
        archive_visdata(
            ms=self.test_ms,
            fieldid=0,
            ddid=0,
            scan=1,
            correlation = 'XX,YY',
            column='DATA',
            compressionrank=1,
            outfilename=self.compressed_file,
            autocorrelation=False,
            weightcr=2,
            flagvalue=0.5 + 0.02j,
            antlist=[0,1,2],
            decorrelation=None
            
        )
        self.assertTrue(os.path.exists(f"zarr-output/{self.compressed_file}"))

        # Step 2: Decompress the data
        decompress_visdata(
            zarr_path=f"zarr-output/{self.compressed_file}",
            output_column='DATA',
            output_ms="decompressed.ms"
            
        )
        self.assertTrue(os.path.exists("decompressed.ms"))

if __name__ == "__main__":
    unittest.main()
