from decimal import Decimal
from io import BytesIO
import os
import sqlite3
from typing import List, Callable, Dict
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from . import settings
from . import aws


# Paths to directories for different file types
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_local_blend_ids(path: str) -> List[str]:
    """Get all of the blend IDs contained in the set
    Either `path` or `set_id` must be given.
    :param path: Path to blends
    :param set_id: Set containing blend ids
    :return: List of blend IDs
    """
    blend_ids = [f.split(".")[0] for f in os.listdir(path)]
    return blend_ids


def get_blend_ids(set_id: int = None) -> List[str]:
    """Get all of the blend IDs contained in the set

    Either `path` or `set_id` must be given.

    :param path: Path to blends
    :param set_id: Set containing blend ids
    :return: List of blend IDs
    """
    assert set_id in [1, 2, 3]
    sql = "SELECT blend_id FROM blends"
    if set_id is not None:
        sql += " WHERE set_id={};".format(set_id)

    path = os.path.join(__ROOT__, "testing", "lookup.db")
    connect = sqlite3.connect(path)
    cursor = connect.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    connect.commit()
    connect.close()

    blend_ids = np.array(result)[:, 0]
    return blend_ids


def get_blend(blend_id: str, path: str = None):
    """Load a blend

    :param blend_id: ID of the blend
    :param path: Path to the blend. If `path` is `None`
        then the blend is loaded from AWS
    :return:
    """
    blend_id = "{}.npz".format(blend_id)
    if path is None:
        client = aws.get_client("s3")
        result = client.get_object(Bucket="scarlet-blends", Key=blend_id)
        data = np.load(BytesIO(result["Body"].read()), allow_pickle=True)
    else:
        data = np.load(os.path.join(path, blend_id))
    return data


def get_branches() -> List[str]:
    """Load all of the branches that have been processed

    :return: List of the branches
    """
    table = aws.get_table("scarlet_branches")
    result = [item["branch"] for item in table.scan()["Items"]]
    return result


def update_merged_branches() -> None:
    import git

    repo = git.Repo()

    commits = [c for c in repo.iter_commits(merges=True)]
    messages = [c.message.split("\n")[0] for c in commits]
    branches = [(m.split("pmelchior/")[1]) for m in messages if "pmelchior" in m][::-1]
    table = aws.get_table("scarlet_merged")
    with table.batch_writer() as batch:
        for idx, branch in enumerate(branches):
            batch.put_item(
                Item={"branch": branch, "merge_order": idx,}
            )


def save_branch(branch: str) -> None:
    """Append a new branch to the branches list

    :param branch: The branch to add to the list
    """
    table = aws.get_table("scarlet_branches")
    with table.batch_writer() as batch:
        batch.put_item(
            Item={"branch": branch,}
        )


def get_measurement_id(measurement, blend_id: str) -> str:
    return "{},{}".format(blend_id, measurement["source_id"])


def save_measurements(
    measurements: List[Dict], set_id: int, branch: str, blend_id: str
) -> None:
    assert set_id in [1, 2, 3]
    table = aws.get_table("scarlet_set{}".format(set_id))
    with table.batch_writer() as batch:
        for measurement in measurements:
            meas_id = get_measurement_id(measurement, blend_id)
            item = {
                "branch": branch,  # primary partition key
                "meas_id": meas_id,  # primary sort key
            }
            item.update(
                {
                    key: Decimal(str(meas))
                    if isinstance(meas, np.floating)
                    else int(meas)
                    for key, meas in measurement.items()
                }
            )
            batch.put_item(Item=item)


def get_object_name(branch, blend_id):
    return "{}/{}.png".format(branch, blend_id)


def save_residual(residual_img, blend_id: str, branch: str):
    from tempfile import NamedTemporaryFile

    # Writing to AWS fails if the temporary file is still open,
    # so we are forced to close the temporary file without deleting it,
    # then manually delete ourselves.
    fp = NamedTemporaryFile("wb", delete=False)
    plt.savefig(fp, bbox_inches="tight")
    fp.close()
    # Save the image to AWS S3
    aws.upload_file(fp.name, "scarlet-residuals", get_object_name(branch, blend_id))
    # Delete the temporary file
    os.remove(fp.name)


def deblend_and_measure(
    set_id: int = None,
    branch: str = None,
    data_path: str = None,
    save_records: bool = False,
    save_residuals: bool = False,
    plot_residuals: bool = False,
    deblender: Callable = None,
    verbose: bool = False,
) -> np.rec.recarray:
    """Deblend an entire test set and store the measurements

    :param set_id: ID of the set to analyze.
        This is only needed if `data_path` is `None` and
        `save` is `True`.
    :param branch: The scarlet branch to test
        (only needed if `save_records` or `save_residuals` is `True`)
    :param data_path: The path to the blend data. If no `data_path is specified
        then __BLEND_PATH__ is used.
    :param save_records: Whether or not to save the measurements records.
    :param save_residuals: Whether or not to save the residual plots
        (only necessary when `plot_residuals=True`).
    :param plot_residuals: Whether or not to plot the residuals.
    :param deblender: The function to use to deblend. This function should only take
        1 argument:

        * `data` The data from the npz file for the blend.

        The function should return a tuple with the following three items:

        * `measurements`: The measurement dictionary entry for the blend
        * `observation`: The observation used for deblending.
        * `sources`: The deblended source models.

    :return: The measurement `records` for each blend.
    """
    if data_path is None:
        blend_ids = get_blend_ids(set_id)
    else:
        blend_ids = get_local_blend_ids(data_path)

    # Use the default `scarlet_extensions` `deblend` if the user hasn't specified their own
    if deblender is None:
        # import here to avoid circular dependence
        from . import deblend

        deblender = partial(
            deblend.deblend, max_iter=settings.max_iter, e_rel=settings.e_rel,
        )

    # If this is the master branch then update the list of merged branches
    if branch == "master" and save_records:
        update_merged_branches()

    # Deblend the scene
    all_measurements = []

    num_blends = len(blend_ids)
    for bidx, blend_id in enumerate(blend_ids):
        if verbose:
            print("blend {} of {}: {}".format(bidx, num_blends, blend_id))
            print(blend_id)
        data = get_blend(blend_id, data_path)
        measurements, observation, sources = deblender(data)
        for m in measurements:
            m["blend_id"] = blend_id
        if save_records:
            save_measurements(measurements, set_id, branch, blend_id)
        all_measurements += measurements

        if plot_residuals or save_residuals:
            import scarlet.display as display

            images = observation.data
            norm = display.AsinhMapping(
                minimum=np.min(images), stretch=np.max(images) * 0.055, Q=10
            )
            fig = display.show_scene(
                sources,
                observation,
                show_model=False,
                show_observed=True,
                show_rendered=True,
                show_residual=True,
                norm=norm,
            )
            plt.suptitle(branch, y=1.05)

            if save_residuals:
                save_residual(fig, blend_id, branch)
            else:
                plt.show()

    # Save the branch if all of the measurements were saved successfully
    if save_records:
        save_branch(branch)

    # Combine all of the records together
    _records = [tuple(m.values()) for m in all_measurements]
    keys = tuple(all_measurements[0].keys())
    records = np.rec.fromrecords(_records, names=keys)
    return records
