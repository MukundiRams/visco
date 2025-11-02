import xarray as xr
import numpy as np
import dask.array as da
import dask
from dask import delayed
import os
import shutil
from daskms import xds_to_table
import visco
from tqdm import tqdm
import sys

log = visco.get_logger(name="VISCO")


class UnifiedProgressBar:
    def __init__(self):
        self.pbar = None
        self.total_steps = 0
        self.current_step = 0

    def start_progress(self, total_steps, initial_desc="Starting..."):
        if self.pbar is not None:
            self.pbar.close()
        self.total_steps = total_steps
        self.current_step = 0
        self.pbar = tqdm(
            total=total_steps,
            desc=initial_desc,
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}",
        )

    def update_step(self, description=None):
        self.current_step += 1
        if self.pbar is not None:
            self.pbar.update(1)
            if description:
                self.pbar.set_description(description)

    def close(self):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


progress = UnifiedProgressBar()


def write_subtable(zarr_path: str, msname: str, group: str):
    """
    Parameters
    ----------
    zarr_path : str
        Path to the Zarr store.
    msname : str
        Path to the output Measurement Set.
    group : str
        Group (subtable) name to write.

    Returns
    -------
    Delayed
        Dask delayed task for writing the subtable.
    """
    ds = xr.open_zarr(zarr_path, group=group)

    if "ROWID" in ds.coords:
        ds = ds.reset_coords("ROWID", drop=True)

    write = xds_to_table(ds, f"{msname}::{group}")

    return write


def list_subtables(zarr_path: str):
    """
    List all subtables (groups) in the Zarr store.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr store.

    Returns
    -------
    list
        List of subtable names.
    """
    return [
        f for f in os.listdir(zarr_path) if os.path.isdir(os.path.join(zarr_path, f))
    ]


def unstack_vis(vis_reconstructed, nrows):
    """Return list of blocks each with shape (nrows, nchan)."""
    if isinstance(vis_reconstructed, da.Array):
        nstack = int(vis_reconstructed.shape[0] // nrows)
        return [
            vis_reconstructed[i * nrows : (i + 1) * nrows, :] for i in range(nstack)
        ]
    else:
        nstack = vis_reconstructed.shape[0] // nrows
        return list(np.split(vis_reconstructed, nstack, axis=0))


def reconstruct_vis(U: da.Array, S: da.Array, Vt: da.Array) -> da.Array:
    """
    Reconstruct the visibility data using the SVD components.

    Parameters
    ----------
    U : dask.array.Array
        Left singular vectors (time, mode).
    S : dask.array.Array
        Singular values (mode,).
    Vt : dask.array.Array
        Right singular vectors (mode, channel).

    Returns
    -------
    dask.array.Array
        Reconstructed visibility data (time, channel).
    """
    if S.ndim == 2:
        S = S[:, 0]

    S_reshaped = S.reshape((1, S.shape[0]))
    U_mult_S = U * S_reshaped

    return U_mult_S @ Vt


def construct_main_ds(zarr_path: str, column: str, batch_size: int):
    """
    Construct the full main table dataset.

    Parameters
    ------
    zarr path (str)
        The path to the zarr store.

    column (str)
        The column in which the compressed data is stored in in the zarr store.

    Returns
    -----
    maintable dataset (xarray dataset)
    """
    progress.update_step("Loading main table and antenna data")
    maintable = xr.open_zarr(zarr_path, group="MAIN", consolidated=True)
    antennas = xr.open_zarr(zarr_path, group="ANTENNA", consolidated=True)
    antnames = antennas.NAME.values

    ant1 = maintable.ANTENNA1.values
    ant2 = maintable.ANTENNA2.values
    data_shape = maintable.DATA.shape
    rowid = maintable.coords["ROWID"].values
    chunks = maintable.DATA.chunks

    reconstructed_data = da.zeros(data_shape, dtype=maintable.DATA.dtype, chunks=chunks)

    baselines = list_subtables(f"{zarr_path}/MAIN/{column}")

    all_baseline_tasks = []

    for baseline_idx, baseline in enumerate(baselines):
        progress.update_step(f"Processing baseline {baseline_idx + 1}/{len(baselines)}")

        correlations = list_subtables(f"{zarr_path}/MAIN/{column}/{baseline}")
        ant1_name, ant2_name = baseline.split("&")
        try:
            ant1_idx = np.where(antnames == ant1_name)[0][0]
            ant2_idx = np.where(antnames == ant2_name)[0][0]
        except IndexError:
            log.warning(f"Baseline {baseline} not found in ANTENNA table. Skipping.")
            continue

        baseline_mask = (ant1 == ant1_idx) & (ant2 == ant2_idx)
        row_indices = np.where(baseline_mask)[0]
        nrows = row_indices.size
        corr_indices = {"XX": 0, "XY": 1, "YX": 2, "YY": -1}

        if not correlations:
            continue

        # build tasks for this baseline
        for corr_name in correlations:
            components = xr.open_zarr(
                f"{zarr_path}/MAIN/{column}/{baseline}/{corr_name}"
            )
            U = components.U.data
            S = components.S.data
            Vt = components.WT.data

            vis_reconstructed = delayed(reconstruct_vis)(U, S, Vt)
            all_baseline_tasks.append(
                (vis_reconstructed, row_indices, corr_name, corr_indices, nrows)
            )

    # process baselines in batches to avoid memory issues
    progress.update_step(f"Computing {len(all_baseline_tasks)} reconstruction tasks")

    reconstructed_data_np = np.zeros(data_shape, dtype=maintable.DATA.dtype)

    # process in batches to manage memory
    for batch_start in range(0, len(all_baseline_tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(all_baseline_tasks))
        batch_tasks = all_baseline_tasks[batch_start:batch_end]

        # compute this batch
        vis_list = [task[0] for task in batch_tasks]
        vis_reconstructed_batch = dask.compute(*vis_list)

        # assign batch results to numpy array
        for idx, (_, row_indices, corr_name, corr_indices, nrows) in enumerate(
            batch_tasks
        ):
            vis_reconstructed = vis_reconstructed_batch[idx]

            # if correlation_optimized used
            if corr_name == "diagonals":
                parts = unstack_vis(vis_reconstructed, nrows)
                reconstructed_data_np[row_indices, :, 0] = parts[0]
                reconstructed_data_np[row_indices, :, 3] = parts[1]
            elif corr_name == "offdiagonals":
                parts = unstack_vis(vis_reconstructed, nrows)
                reconstructed_data_np[row_indices, :, 1] = parts[0]
                reconstructed_data_np[row_indices, :, 2] = parts[1]
            else:
                corr_idx = corr_indices[corr_name]
                reconstructed_data_np[row_indices, :, corr_idx] = vis_reconstructed

        progress.update_step(f"Assembled batch {batch_end}/{len(all_baseline_tasks)}")

    reconstructed_data = da.from_array(reconstructed_data_np, chunks=chunks)

    progress.update_step("Loading and processing flag data")

    flags_ds = xr.open_zarr(zarr_path, group="FLAGS", consolidated=True)
    flags_length = data_shape[0] * data_shape[1] * data_shape[2]
    flags = np.unpackbits(flags_ds.FLAGS.values, count=flags_length)
    flags = flags.reshape(data_shape)

    flag_row_ds = xr.open_zarr(zarr_path, group="FLAGS_ROW", consolidated=True)
    flags_row = np.unpackbits(flag_row_ds.FLAGS_ROW.values, count=data_shape[0])

    if "WEIGHT_SPECTRUM" in list_subtables(f"{zarr_path}"):
        progress.update_step("Processing weight spectrum")

        weights = xr.open_zarr(f"{zarr_path}/WEIGHT_SPECTRUM", consolidated=True)
        weights_reconstructed = np.dot(weights.U.values, np.diag(weights.S.values))
        weights_expanded = np.expand_dims(weights_reconstructed, axis=-1)
        final_weights = np.tile(weights_expanded, (1, 1, data_shape[2]))

        maintable = maintable.assign(
            **{
                "WEIGHT_SPECTRUM": xr.DataArray(
                    da.from_array(final_weights, chunks=chunks),
                    dims=("row", "chan", "corr"),
                    coords={"ROWID": ("row", rowid)},
                ),
                "SIGMA_SPECTRUM": xr.DataArray(
                    da.from_array(final_weights, chunks=chunks),
                    dims=("row", "chan", "corr"),
                    coords={"ROWID": ("row", rowid)},
                ),
            }
        )

    progress.update_step("Finalizing main table dataset")
    maintable = maintable.assign(
        **{
            "DATA": xr.DataArray(
                reconstructed_data,
                dims=("row", "chan", "corr"),
                coords={"ROWID": ("row", rowid)},
            ),
            "FLAG": xr.DataArray(
                da.from_array(flags, chunks=chunks),
                dims=("row", "chan", "corr"),
                coords={"ROWID": ("row", rowid)},
            ),
            "FLAG_ROW": xr.DataArray(
                da.from_array(flags_row, chunks=chunks[0]),
                dims=("row",),
                coords={"ROWID": ("row", rowid)},
            ),
        }
    )

    return maintable


def open_dataset(
    zarr_path: str,
    column: str = "COMPRESSED_DATA",
    group: str = None,
    batch_size: int = 50,
):
    """ "
    Open the zarr store in a MSv2 format including the SVD components.

    Parameters
    ------
    zarr path (str)
        The path to the zarr store.

    column (str)
        The column in which the compressed data is stored in in the zarr store.

    group (str)
        MS group/subtable to open. If none, the main table is opened. Default is None.

    Returns
    -----
    dataset (xarray dataset)
    """
    if group is None:
        maintable = construct_main_ds(
            zarr_path=zarr_path, column=column, batch_size=batch_size
        )
        return maintable
    else:
        ds = xr.open_zarr(zarr_path, group=group, consolidated=True)
        return ds


def write_datasets_to_ms(zarr_path: str, msname: str, column: str, batch_size: int):
    """ "
    Write all the datasets to the Measurement set.

    Parameters
    ------
    zarr path (str)
        The path to the zarr store.

    msname (str)
        The name of the output MS.

    column (str)
        The column in which the compressed data is stored in in the zarr store.

    batch_size (int)
        Number of baseline reconstruction tasks to process in a batch.

    Returns
    -----
    None
    """
    baselines = list_subtables(f"{zarr_path}/MAIN/{column}")
    b0 = baselines[0]
    correlations = list_subtables(f"{zarr_path}/MAIN/{column}/{b0}")
    zarr_folders = list_subtables(zarr_path)
    non_folders = ["MAIN", "FLAGS", "FLAG_ROW", "WEIGHT_SPECTRUM"]
    subtable_count = len([f for f in zarr_folders if f not in non_folders])

    # total steps: setup + baselines + assembly + flag processing + weight spectrum + finalize + write main + subtables + completion
    total_steps = (
        1
        + len(baselines)
        + 1
        + 3
        + 1
        + subtable_count
        + 1
        + len(baselines) * len(correlations) / batch_size
    )

    progress.start_progress(total_steps, "Starting MS decompression...")

    if os.path.exists(msname):
        progress.update_step("Removing existing MS")
        shutil.rmtree(msname)

    maintable = construct_main_ds(
        zarr_path=zarr_path, column=column, batch_size=batch_size
    )

    progress.update_step("Writing main table to Measurement Set...")
    write_main = xds_to_table(maintable, f"{msname}")

    progress.update_step("Computing main table (this may take a while)...")
    from dask.diagnostics import ProgressBar

    with ProgressBar():
        dask.compute(write_main)

    tasks = []
    for folder in zarr_folders:
        if folder in non_folders:
            continue
        progress.update_step(f"Preparing subtable: {folder}")
        task = write_subtable(zarr_path, msname, folder)
        tasks.append(task)

    if tasks:
        progress.update_step("Writing all subtables...")
        dask.compute(*tasks)

    progress.update_step("✅Measurement Set creation completed successfully!")
    progress.close()
