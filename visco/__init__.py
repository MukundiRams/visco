import logging
import os
from importlib import metadata
from omegaconf import OmegaConf
from dask.distributed import Client, LocalCluster

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


def setup_dask_client(memory_limit:str,nworkers:int,nthreads:int,direct_to_workers:bool,
                      silence_logs:bool,dashboard_addr:str=None,host_addr:str=None)->Client:
    """
    Set up a Dask client based on system resources.
    """
    if silence_logs == True:
        logging.getLogger("distributed").setLevel(logging.ERROR)
        logging.getLogger("dask").setLevel(logging.ERROR)
        logging.getLogger("asyncio").setLevel(logging.ERROR)
    elif silence_logs == False:
        logging.getLogger("distributed").setLevel(logging.INFO)
        logging.getLogger("dask").setLevel(logging.INFO)
        logging.getLogger("asyncio").setLevel(logging.INFO)
        
    else:
        logging.getLogger("distributed").setLevel(logging.WARNING)
        logging.getLogger("dask").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    host_address = host_addr or os.environ.get("DASK_SCHEDULER_ADDRESS")
    if host_addr is not None:
        print(f"Connecting to Dask scheduler at {host_address}...")
        client = Client(host_address)
        
        try:
            print("Dask client created.")
            print("  dashboard_link:", getattr(client, "dashboard_link", None))
        
        except Exception as e:
            logging.exception("Error printing dask debug info: %s", e)
        
    else:
        print("Setting up local Dask cluster...")
        cluster = LocalCluster(
            n_workers=nworkers, 
            threads_per_worker=nthreads,
            memory_limit=memory_limit,
            processes=True, 
            asynchronous=False, 
            silence_logs=silence_logs,
            dashboard_address=dashboard_addr
        )
        
        client = Client(cluster,direct_to_workers=direct_to_workers
                        )
        
        try:
            print("Dask client created.")
            print("  scheduler:", cluster.scheduler_address)
            print("  dashboard_link:", getattr(client, "dashboard_link", None))
            print("  cluster.dashboard_address:", getattr(cluster, "dashboard_address", None))
        
        except Exception as e:
            logging.exception("Error printing dask debug info: %s", e)
        
    try:
        client.wait_for_workers(nworkers, timeout=10)
    except TimeoutError:
        logging.warning("Timeout waiting for workers; continuing with what we have.")

    return client
