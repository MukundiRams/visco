from doctest import OutputChecker
import glob
import os

import click
from omegaconf import OmegaConf
from scabha.basetypes import File
from scabha.schema_utils import clickify_parameters, paramfile_loader
import zarr

import visco
from visco import BIN, get_logger
from visco import decompress

log = get_logger(BIN.decompression)

command = BIN.decompression

thisdir = os.path.dirname(__file__)
decompression_params = glob.glob(f"{thisdir}/*.yaml")
decompression_files = [File(item) for item in decompression_params]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, decompression_files)[command]


@click.command(command)
@click.version_option(str(visco.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    zarr_path = opts.zarr_path
    output_column = opts.output_column
    ms = opts.ms
    
    decompress.decompress_visdata(zarr_path, output_column, ms)