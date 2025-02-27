import dask.array as da
from daskms import xds_from_ms, xds_to_table, xds_from_table
import daskms
import dask
from casacore.tables import table
import logging
from casacore.tables import table

logging.getLogger("daskms").setLevel(logging.ERROR)
log = logging.getLogger("visco")


# def add_column(msfile, outcol, write_data=None):
#     """
#     Add a column to the measurement set and populate it with the given data.
    
#     Parameters
#     ----
#     msfile: str
#         - Path to measurement set.
    
#     outcol: str
#         - Column name to be added to the measurement set.
    
#     write_data:
#         - The data to populate the added column with.
    
#     Returns
#     ----
#     bool:
#         True if column is added successfully.
#     """
    
#     try:
#         #access the measurement set
#         table = xds_from_table(msfile)[0]
#         num_rows = len(table.ROWID.data.compute())
#         data_shape = table.DATA.shape
#         data_chunks = table.DATA.chunks
#         zero_data = da.zeros(data_shape, chunks=data_chunks)
        
#         if outcol in table:
#             print(f"Data column {outcol} already exists, will update it now.")
#         else:
#             print(f"Data column {outcol} doesn't exists, will create it now and populate it.")
        
#         #check if write_data is provided and has the correct shape
#         if write_data is not None:
#             if write_data.shape != data_shape:
#                 raise ValueError(f"Data to be written with shape {write_data.shape} does not match table data shape {data_shape}.")
#             table[outcol] = (("row", "chan", "corr"), write_data)
#         else:
#             table[outcol] = (("row", "chan", "corr"), zero_data)
        
#         main_table = daskms.Dataset(
#             table, coords={"ROWID": ("row", da.arange(num_rows, chunks=(data_chunks[0],)))}
#         )
        
#         write_main = xds_to_table(main_table, msfile, columns=[outcol])
#         dask.compute(write_main)
        
#         return True
    
#     except Exception as e:
#         print('the error is here.')
#         print(f"An error occurred: {e}")
        
#         return False



# def delete_column(msfile, colname):
#     """
#     Delete a column from the measurement set.
    
#     Parameters
#     ----
#     msfile: str
#         - Path to measurement set.
    
#     colname: str
#         - Column name to be deleted from the measurement set.
    
#     Returns
#     ----
#     bool:
#         True if column is deleted successfully, False otherwise.
#     """
    
#     try:
        
#         table = table(msfile)
        
#         if colname in table.colnames():
#             table.removecols(colname)
#             print(f"Data column {colname} exists, deleting it now.")
#         else:
#             print(f"Data column {colname} does not exist in the measurement set.")
#             return False
    
        
#         updated_table = xds_from_table(msfile)[0]
        
#         if colname in updated_table:
#             print(f"Column {colname} still exists in the measurement set.")
#             return False
#         else:
#             print(f"Column {colname} successfully deleted.")
#             return True
        
    
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return False

def add_column(msfile: str, outcol: str, write_data: da.Array = None):
    """
    Add a column to a measurement set and populate it with the given data.
    
    Parameters
    ----
    msfile: str
        - Path to measurement set.
    
    outcol: str
        - Column name to be added to the measurement set.
    
    write_data:
        - The data to populate the added column with.
    
    Returns
    ----
    bool:
        True if column is added successfully.
    """
    
    try:
        #access the measurement set
        table = xds_from_table(msfile)[0]
        num_rows = len(table.ROWID.data.compute())
        data_shape = table.DATA.shape
        data_chunks = table.DATA.chunks
        zero_data = da.zeros(data_shape, chunks=data_chunks)
        
        if outcol in table:
            log.info(f"Data column {outcol} already exists, will update it now!")
            # delete_column(msfile,outcol)
        else:
            log.info(f"Data column {outcol} doesn't exist, will create it and populate it now!")
        
        #check if write_data is provided and has the correct shape
        if write_data is not None:
            if write_data.shape != data_shape:
                raise ValueError(f"Data to be written with shape {write_data.shape} does not match table data shape {data_shape}.")
            write_data = da.rechunk(write_data, chunks=data_chunks)
            table[outcol] = (("row", "chan", "corr"), write_data)
        else:
            table[outcol] = (("row", "chan", "corr"), zero_data)
        
        main_table = daskms.Dataset(
            table, coords={"ROWID": ("row", da.arange(num_rows, chunks=(data_chunks[0],)))}
        )
        
        write_main = xds_to_table(main_table, msfile, columns=[outcol])
        dask.compute(write_main)
        
        return True
    
    except Exception as e:
        log.error(f"An error occurred while adding column: {e}")
        return False

def delete_column(msfile: str, colname: str):
    """
    Delete a column from the measurement set.
    
    Parameters
    ----
    msfile: str
        - Path to measurement set.
    
    colname: str
        - Column name to be deleted from the measurement set.
    
    Returns
    ----
    bool:
        True if column is deleted successfully, False otherwise.
    """
    
    try:
        # Open the table directly with casacore
        with table(msfile, readonly=False,lockoptions='auto') as tb:
            if colname in tb.colnames():
                tb.removecols(colname)
                log.info(f"Data column {colname} exists, deleting it now!")
            else:
                log.info(f"Data column {colname} does not exist in the measurement set.")
                return False

        # Verify deletion
        with table(msfile, readonly=True) as tb:
            if colname not in tb.colnames():
                log.info(f"Column {colname} successfully deleted.")
                return True
            else:
                log.error(f"Column {colname} still exists in the measurement set after attempted deletion.")
                return False

    except Exception as e:
        log.error(f"An error occurred while deleting column: {e}")
        return False

