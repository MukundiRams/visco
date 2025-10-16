.. _tutorials:

Tutorials
#########

This tutorial demonstrates how to use **visco** for compressing and decompressing radio interferometric data. We generate a Measurement Set (MS) and synthetic visibilities using `simms` and MeqTrees, respectively.

Point Source Simulation at Phase Centre with KAT-7
==================================================

- The telescope is simulated with a 1-hour observation, using an integration time of 2 seconds.
- The starting frequency is 1.4 GHz, with 64 channels, each with a width of 100 kHz.
- The source is unpolarized with a total intensity of 1 Jy.
- The full rank of the data is min(timeslots,channels) = 64.

Original image produced using WSClean:

.. image:: kat7-sim-dirty.png
   :alt: Original Image
   :width: 400px
   :align: center

**Statistics:**

We use CARTA to open the images and measure the statistics.

Peak flux:  :math: `1.000429391861 \times 10^{0}` Jy/beam 
RMS: :math: 1.439807312603 \times 10^{-1} Jy/beam.
SNR: 6.9484

**Disk usage:**
The MS occupies 228 MB of disk storage.


Compressing the Visibility Data
------------------------------

To compress the visibility data, run:

::

   visco compressms -ms kat7-sim.ms/ -zs kat7-sim.zarr -col DATA -corr XX,XY,YX,YY -cr 1 -nw 8 -nt 1 -ml 16GB -da 2727 -csr 3600

where:

- `-ms` gives the path to the measurement set,
- `-zs` specifies the output Zarr store,
- `-col` specifies the column containing the visibility data,
- `-corr` defines the correlations to compress,
- `-cr` is the desired compression rank,
- `-nw` sets the number of Dask workers,
- `-nt` specifies the number of threads per worker,
- `-ml` sets the memory limit,
- `-da` is the dashboard address,
- `-csr` is the chunk size along the row.

We compressed the data with a compression rank of 1, retaining only the components corresponding to the first singular value.

After compression, the data is stored as `meerkat-sim.zarr`.

Decompressing the Compressed Data
---------------------------------

To decompress back into an MS for imaging, run:

::

   visco decompressms -zp kat7-sim.zarr/ -ms kat7-sim-decompressed.ms

where:

- `-zp` provides the path to the Zarr store containing the compressed data,
- `-ms` sets the output MS file.

After decompression, the output image (produced using WSClean) is:

.. image:: kat7-sim-decompressed-dirty.png
   :alt: Image after compressing the visibility data
   :width: 400px
   :align: center

**Statistics:**

Peak flux: :math: 1.000427842140 \times 10^{0} Jy/beam 
RMS: :math: 1.439824590852 \times 10^{-1} Jy/beam.
SNR: 6.9483

**Disk usage:**  
The compressed Zarr store uses only 15 MB of disk space.

Combining Correlations
----------------------

Compressing the visibility data using SVD is computationally expensive because the operation must be performed for each baseline and correlation product. To reduce computational cost, you can choose to combine correlations. This approach optimizes compression by grouping the XX & YY and XY & YX correlations together.

Although our simulation so far includes an unpolarized source where the XY and YX correlations contribute minimally, we can still test this approach. To do so, simply add the ``--correlation-optimized`` flag:

::

   visco compressms -ms kat7-sim.ms/ -zs kat7-sim.zarr -col DATA -corr XX,XY,YX,YY -cr 1 -nw 8 -nt 1 -ml 16GB -da 2727 -csr 3600 --correlation-optimized

This compression produces the following image:

.. image:: kat7-sim-corropt-dirty.png
   :alt: Image after compressing the visibility data by combining correlations
   :width: 400px
   :align: center

**Statistics:**

Peak flux: :math: 1.000429391861  \times 10^{0} Jy/beam 
RMS: :math: 1.439828319666 \times 10^{-1} Jy/beam.
SNR: 6.9483

**Disk usage:**
This correlation optimization also reduces storage requirements, with only 9 MB required.





