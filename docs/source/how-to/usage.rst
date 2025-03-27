.. _usage:
.. role:: raw-math(raw)
    :format: latex html

Running Visco
==================
Visco has two applications (or scripts). The first one, ``archival``, is for compressing radio interferometry data (or visibilities) in a `Measurement Set (MS) <https://casa.nrao.edu/Memos/229.html>`_  using Singular Value Decomposition (SVD). This application stores the compressed data in a ZARR file. Use ``archival --help`` for a general overview of the application and defintion of the parameters.

   
The ``archival`` application has two options. The first one, simple SVD, compresses the visibility data using the same rank or number of singular values across all the baselines. 
For instance, if we wanted to compress radio interferometry data in an MS using simple SVD, we would run the command:

.. code-block:: bash

    archival --fieldid 0 --ddid 0 --scan 1 --corr XX,YY --column DATA --compressionrank 3 --outfilename compressed-data.zarr msfile.ms

Note that ``--compressionrank`` is the rank or number of singular values that are retained across all the baselines duiring the compression. ``corr`` are the correlations to compress and store. The input MS is ``msfile.ms`` and the output ZARR file is ``compressed-data.zarr``.

The other option is Baseline-Dependent SVD, which applies variable compression rank based on the baseline. For example, if we wanted to preserve at least :raw-math:`98\%` of the signal on each baseline, we would run the following command:

.. code-block:: bash

    archival --fieldid 0 --ddid 0 --scan 1 --corr XX,YY --column DATA --decorrelation 0.98 --outfilename compressed-data.zarr msfile.ms


The other application is ``decompression``. This decompresses the archived data from a ZARR file to a MS. To decompress the archived data in the compressed-data.zarr file, one should run:
   
.. code-block:: bash

    decompression --zarr-path compressed-data.zarr --outcol DATA --ms decompressed-data.ms
