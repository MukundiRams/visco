import glob
import os

import click
from omegaconf import OmegaConf
from scabha.basetypes import File
from scabha.schema_utils import clickify_parameters, paramfile_loader

import visco
from visco import BIN, get_logger
from visco import archive

log = get_logger(BIN.archival)

command = BIN.archival

thisdir = os.path.dirname(__file__)
archival_params = glob.glob(f"{thisdir}/*.yaml")
archival_files = [File(item) for item in archival_params]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, archival_files)[command]



@click.command(command)
@click.version_option(str(visco.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    ms = opts.ms
    fieldid = opts.fieldid
    ddid = opts.ddid
    scan = opts.scan
    column = opts.column
    autocorrelation = opts.autocorrelation
    antlist = opts.antlist
    flagvalue = opts.flagvalue
    correlation = opts.correlation
    outfilename = opts.outfilename
    compressionrank = opts.compressionrank
    decorrelation = opts.decorrelation
    archive.archive_visdata(ms, correlation, fieldid, ddid, scan,
                      column, outfilename, compressionrank,
                      autocorrelation, decorrelation, antlist, flagvalue)
    