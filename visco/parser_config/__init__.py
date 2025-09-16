import click
import visco


@click.group(help="A tool for compressing radio interferometric data using"
             " lossy Singular Value Decomposition (SVD) techniques.\n\n"
             "Also includes utilities for decompressing the data back to a"
             " Measurement Set (MS) format.\n\n"
             "Author: Mukundi Ramanyimi")

@click.version_option(str(visco.__version__))
def cli():
    pass

def add_commands():
    # Importing the commands in a function to avoid a circular import error
    from .compressms import compressrunit
    from .decompressms import decompressrunit
    
add_commands()
