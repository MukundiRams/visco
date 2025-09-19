.. _usage:
.. role:: raw-math(raw)
    :format: latex html

Running Visco
==================
Visco has two applications (or scripts). The first one, ``compressms``, is for compressing radio interferometric data (or visibilities) in a `Measurement Set (MS) <https://casa.nrao.edu/Memos/229.html>`_  using Singular Value Decomposition (SVD). This application stores the compressed data in a Zarr file. Use ``visco compressms --help`` for a general overview of the application and defintion of the parameters.

   
The ``compressms`` application has two options. The first one, simple SVD, compresses the visibility data using the same rank or number of singular values across all the baselines. 
For instance, if we wanted to compress radio interferometry data in an MS using simple SVD, we would run the command:

.. code-block:: bash

    visco compressms --ms yourms.ms --zarrstore youroutputzarr.zarr --chunk-size-row 10000 --fieldid 0 --ddid 0 --scan 1 --correlation XX,YY --column DATA --compressionrank 3 

Note that ``--compressionrank`` is the rank or number of singular values that are retained across all the baselines duiring the compression. ``correlation`` are the correlations to compress and store. The input MS is ``yourms.ms`` and the output ZARR file is ``youroutputzarr.zarr``. ``chunk-size-row`` is the number of rows per chunk in the output Zarr file.

The other option is Baseline-Dependent SVD, which applies variable compression rank based on the baseline. For example, if we wanted to preserve at least :raw-math:`98\%` of the signal on each baseline, we would run the following command:

.. code-block:: bash

    visco compressms --ms yourms.ms --zarrstore youroutputzarr.zarr --chunk-size-row 10000 --fieldid 0 --ddid 0 --scan 1 --correlation XX,YY --column DATA --decorrelation 0.98

There are other parameters that can be used to customize the compression. Please refer to the help message of the application for more details.

The other application is ``decompressms``. This decompresses the compressed data from a Zarr file back to a MS. To decompress the compressed data in the Zarr file, one should run:
   
.. code-block:: bash

    visco decompressms --zarr-path youroutputzarr.zarr --ms decompressed-data.ms
