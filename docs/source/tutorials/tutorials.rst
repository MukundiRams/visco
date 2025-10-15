Tutorials
=========

This tutorial demonstrates how to use **visco** to compress and decompress radio interferometry visibility data using Singular Value Decomposition (SVD).

Installation
------------

Install visco using pip:

::

   pip install visco

Compressing Visibility Data
----------------------------

The compression process reduces the storage requirements of measurement sets while preserving the essential astronomical information.

Basic Compression Command
~~~~~~~~~~~~~~~~~~~~~~~~~

To compress visibility data, execute the following command:

::

   visco compressms -ms kat7-sim.ms/ -zs kat7-sim.zarr -corr XX,XY,YX,YY -cr 1 -nw 8 -nt 1 -ml 16GB -da 2727 -csr 3600

Command Parameters
~~~~~~~~~~~~~~~~~~

The parameters are defined as follows:

- ``-ms``: Path to the input measurement set
- ``-zs``: Output path for the compressed Zarr store
- ``-corr``: Correlation products to compress (XX, XY, YX, YY)
- ``-cr``: Compression rank (number of singular values to retain)
- ``-nw``: Number of Dask workers for parallel processing
- ``-nt``: Number of threads per worker
- ``-ml``: Memory limit per worker
- ``-da``: Dashboard address for monitoring
- ``-csr``: Chunk size along the row dimension

Compression Method
~~~~~~~~~~~~~~~~~~

In this example, we use a compression rank of 1, which retains only the components corresponding to the first singular value. This provides substantial compression while maintaining image fidelity for strong sources.

After compression, the data is stored in the Zarr format as ``kat7-sim.zarr``.

Decompressing Data
------------------

To decompress the Zarr store back into a measurement set for imaging:

::

   visco decompressms -zp kat7-sim.zarr/ -ms kat7-sim-decompressed.ms

Command Parameters
~~~~~~~~~~~~~~~~~~

- ``-zp``: Path to the Zarr store containing compressed data
- ``-ms``: Output measurement set file path

Imaging Results
~~~~~~~~~~~~~~~

After decompression, you can image the data using standard tools such as WSClean. The resulting image is shown below:

.. image:: kat7-sim-decompressed-image.png
   :alt: Image produced after compressing and decompressing the visibility data
   :width: 400px
   :align: center

Image Quality Metrics
~~~~~~~~~~~~~~~~~~~~~

The image produced from the compressed visibility data has the following properties:

- **Peak flux:** 1.249 × 10⁰ Jy/beam
- **RMS noise:** 1.314 × 10⁻¹ Jy/beam
- **Signal-to-Noise Ratio (SNR):**
  
  - Original image: 9.595
  - Compressed image: 9.505

Storage Requirements
~~~~~~~~~~~~~~~~~~~~

- **Original measurement set:** 228 MB
- **Compressed Zarr store:** 15 MB

This represents a compression ratio of approximately **15:1** with minimal loss in image quality.

Optimized Compression with Correlation Combining
-------------------------------------------------

Overview
~~~~~~~~

For unpolarized sources, the cross-correlation products (XY and YX) contribute minimal signal. Combining correlations during compression can significantly improve computational speed and further reduce storage requirements.

Optimized Compression Command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable correlation optimization, add the ``--correlation-optimized`` flag:

::

   visco compressms -ms kat7-sim.ms/ -zs kat7-sim.zarr -corr XX,XY,YX,YY -cr 1 -nw 8 -nt 1 -ml 16GB -da 2727 -csr 3600 --correlation-optimized

Benefits
~~~~~~~~

1. **Computational efficiency:** Reduces processing time by compressing combined correlation products
2. **Storage savings:** Further reduces disk usage compared to standard compression
3. **Maintained fidelity:** Preserves image quality for unpolarized sources

Imaging Results
~~~~~~~~~~~~~~~

The image produced with correlation optimization:

.. image:: kat7-sim-corropt-image.png
   :alt: Image after compressing with correlation optimization
   :width: 400px
   :align: center

Storage Comparison
~~~~~~~~~~~~~~~~~~

- **Original measurement set:** 228 MB
- **Standard compressed Zarr:** 15 MB
- **Optimized compressed Zarr:** 9 MB

The correlation-optimized approach provides a compression ratio of approximately **25:1**, offering substantial storage savings for unpolarized data.

Summary
-------

The **visco** package provides efficient compression of radio interferometry visibility data:

1. **Standard compression** (rank 1): Reduces 228 MB to 15 MB (~15:1 ratio)
2. **Optimized compression** (correlation combining): Reduces 228 MB to 9 MB (~25:1 ratio)
3. **Image quality:** Minimal degradation (SNR: 9.595 → 9.505)
4. **Use case:** Ideal for archiving, data transfer, and processing large interferometric datasets

These compression techniques enable more efficient storage and handling of large-scale radio astronomy data while maintaining scientific integrity.
