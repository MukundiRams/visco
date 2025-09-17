import glob
import os
import click
from omegaconf import OmegaConf
from scabha.basetypes import File
from scabha.schema_utils import clickify_parameters, paramfile_loader
import visco
from visco import BIN, get_logger
from visco import compress_ms
from . import cli

log = get_logger(BIN.compressms)

command = BIN.compressms

thisdir = os.path.dirname(__file__)
compression_params = glob.glob(f"{thisdir}/*.yaml")
compression_files = [File(item) for item in compression_params]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, compression_files)[command]



@cli.command(command)
@click.version_option(str(visco.__version__))
@clickify_parameters(config)
def compressrunit(**kwargs):
    opts = OmegaConf.create(kwargs)
    ms = opts.ms
    fieldid = opts.fieldid
    ddid = opts.ddid
    scan = opts.scan
    zarr_path = opts.zarrstore
    chunk_size_row = opts.chunk_size_row
    consolidated = opts.consolidated
    overwrite = opts.overwrite
    compressor = opts.compressor
    level = opts.level
    correlation = opts.correlation
    corr_opt = opts.correlation_optimized
    column = opts.column
    outcolumn = opts.outcolumn
    use_model_data = opts.use_model_data
    model_data = opts.model_data
    flag_estimate = opts.flagestimate
    decorrelation = opts.decorrelation
    compressionrank = opts.compressionrank
    flagvalue = opts.flagvalue 
    antennas = opts.antennas
    nworkers = opts.nworkers
    nthreads = opts.nthreads
    memory_limit = opts.memory_limit
    direct_to_workers = opts.direct_to_workers
    silence_logs = opts.silence_logs
    dashboard_addr = opts.dashboard_address
    
    compress_ms.compress_full_ms(ms_path=ms, zarr_path=zarr_path, 
                                 consolidated=consolidated,
                                 chunk_size_row=chunk_size_row,
                                 overwrite=overwrite,
                                 compressor=compressor,
                                 nworkers=nworkers,
                                 nthreads=nthreads,
                                 memory_limit=memory_limit,
                                 direct_to_workers=direct_to_workers,
                                 silence_logs=silence_logs,
                                 level=level,
                                 correlation=correlation,
                                 correlation_optimized=corr_opt,
                                 fieldid=fieldid, ddid=ddid,scan=scan,
                                 column=column, outcolumn=outcolumn,
                                 dashboard_addr=dashboard_addr,
                                 use_model_data=use_model_data, model_data=model_data,
                                 flag_estimate=flag_estimate, decorrelation=decorrelation,
                                 compressionrank=compressionrank,flagvalue=flagvalue,
                                 antennas=antennas)
                                  