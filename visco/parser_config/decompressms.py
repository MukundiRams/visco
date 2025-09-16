from doctest import OutputChecker
import glob
import os
import click
from omegaconf import OmegaConf
from scabha.basetypes import File
from scabha.schema_utils import clickify_parameters, paramfile_loader
import visco
from visco import BIN, get_logger
from visco import decompress_ms
from . import cli

log = get_logger(BIN.decompressms)

command = BIN.decompressms

thisdir = os.path.dirname(__file__)
decompressms_params = glob.glob(f"{thisdir}/*.yaml")
decompressms_files = [File(item) for item in decompressms_params]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, decompressms_files)[command]


@cli.command(command)
@click.version_option(str(visco.__version__))
@clickify_parameters(config)
def decompressrunit(**kwargs):
    opts = OmegaConf.create(kwargs)
    zarr_path = opts.zarr_path
    column = opts.column
    ms = opts.ms
    
    decompress_ms.write_datasets_to_ms(zarr_path, ms, column)
    