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

Note that we use dirty images here since the compression can affect the Point Spread Function (PSF). We use CARTA to open the images and measure the statistics:

- **Peak flux:** \(:math:`1.000429391861 \times 10^{0}`\) Jy/beam
- **RMS:** \(:math:`1.439807312603 \times 10^{-1}`\) Jy/beam
- **SNR:** 6.9484


**Disk usage:**
The MS occupies 228 MB of disk storage.


Compressing the Visibility Data
------------------------------

To compress the visibility data, run:

::

   visco compressms -ms kat7-sim.ms/ -zs kat7-sim.zarr -col DATA -corr XX,XY,YX,YY -cr 1 -nw 8 -nt 1 -ml 4GB -da 2727 -csr 3600

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

We compressed the data with a compression rank of 1, retaining only the components corresponding to the first singular value. The compression results are stored as `meerkat-sim.zarr`. In this Zarr store, the data are stored as separate components U,S,V for each correlation and baseline. U,S, and V correspond to the resulting matrices from singular value decomposition (SVD).

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

- **Peak flux:** \(:math:`1.000427842140 \times 10^{0}`\) Jy/beam
- **RMS:** \(:math:`1.439824590852 \times 10^{-1}`\) Jy/beam
- **SNR:** 6.9483


**Disk usage:**  
The compressed Zarr store uses only 15 MB of disk space.

Combining Correlations
----------------------

Compressing the visibility data using SVD is computationally expensive because the operation must be performed for each baseline and correlation product. To reduce computational cost, you can choose to combine correlations. This approach optimizes compression by grouping the XX & YY and XY & YX correlations together.

Although our simulation so far includes an unpolarized source where the XY and YX correlations contribute minimally, we can still test this approach. To do so, simply add the ``--correlation-optimized`` flag:

::

   visco compressms -ms kat7-sim.ms/ -zs kat7-sim.zarr -col DATA -corr XX,XY,YX,YY -cr 1 -nw 8 -nt 1 -ml 4GB -da 2727 -csr 3600 --correlation-optimized

This compression produces the following image:

.. image:: kat7-sim-corropt-dirty.png
   :alt: Image after compressing the visibility data by combining correlations
   :width: 400px
   :align: center

**Statistics:**

- **Peak flux:** \(:math:`1.000429391861 \times 10^{0}`\) Jy/beam
- **RMS:** \(:math:`1.439828319666 \times 10^{-1}`\) Jy/beam
- **SNR:** 6.9483


**Disk usage:**
This correlation optimization also reduces storage requirements, with only 9 MB required for the Zarr store.


Multiple Point Sources with the MeerKAT Telescope
===================================================

Now, lets focus on the MeerKAT telescope. Using the same simulation settings as the ones we used for KAT-7, we simulate 10 point sources. All 10 point sources are unpolarized with total intensity of 1 Jy.

For this simulation, we get this image:

.. image:: meerkat-sim-dirty.png
   :alt: Original image for the MeerKAT simulation
   :width: 400px
   :align: center

**Statistics:**
For the source furthest from the phase centre:

- **Peak flux:** \(:math:`9.556545019150 \times 10^{-1}`\) Jy/beam
- **RMS:** \(:math:`1.629931402737 \times 10^{-2}`\) Jy/beam
- **SNR:** 58.6316

The compression for the MeerKAT telescope is extremely computationally expensive as there are 2016 baselines, so we add the --batch-size flag, which let us decide the number of the baselines to process at the same time. We still choose to compress using the first singular value.

::

   visco compressms -ms meerkat-sim.ms/ -zs meerkat-sim.zarr -col DATA -corr XX,XY,YX,YY -cr 1 -nw 8 -nt 1 -ml 4GB -da 2727 -csr 10000 -bs 200

where:
- `-bs` determine the batch size or the number of baselines to process at once.














