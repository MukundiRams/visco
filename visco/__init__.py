import logging
import os
from importlib import metadata
from omegaconf import OmegaConf

__version__ = metadata.version(__package__)

PCKGDIR = os.path.dirname(os.path.abspath(__file__))

SCHEMADIR = os.path.join(__path__[0], "schemas")


def get_logger(name, level="DEBUG"):

    if isinstance(level, str):
        level = getattr(logging, level, 10)

    format_string = '%(asctime)s-%(name)s-%(levelname)-8s| %(message)s'
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format=format_string,
                        datefmt='%m:%d %H:%M:%S')

    return logging.getLogger(name)


LOG = get_logger("VISCO")

BIN = OmegaConf.create({"visco": "visco",
                        "compressms": "compressms",
                        "decompressms": "decompressms"
                        })


from dask.distributed import Client, LocalCluster


def setup_dask_client(memory_limit:str,nworkers:int,nthreads:int,direct_to_workers:bool):
    """
    Set up a Dask client based on system resources.
    """
    
    cluster = LocalCluster(
        n_workers=nworkers, 
        threads_per_worker=nthreads,
        memory_limit=memory_limit,
        processes=True, 
        asynchronous=False, 
        silence_logs=logging.ERROR,
    )
    
    client = Client(cluster,direct_to_workers=direct_to_workers
                    )
    
    client.wait_for_workers(nworkers)
  
    return client
