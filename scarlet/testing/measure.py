import os
from typing import List, Sequence, Dict, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

from .api import get_branches


def adjacent_values(vals: np.ndarray, q1: int, q3: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get adjacent values for whiskers

    :param vals: The array that is being plotted
    :param q1: The lower quartile
    :param q3: The upper quartile
    :return: lower whisker and upper whisker value
    """
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def measure_blend(
        data: Dict[str, np.ndarray],
        sources: List,
        filters: Sequence[str],
) -> List[Dict[str, float]]:
    """
    Measure all of the fake sources in a single blend

    :param data: The numpy file with blend data
    :param sources: The sources in the blend
    :param filters: The filter name for each band
    :return: List of measurements for each matched source
    """
    import scarlet.measure

    # Extract necessary fields from the data
    centers = data["centers"]
    matched = data["matched"]
    matched_centers = np.array([[m["y"], m["x"]] for m in matched]).astype(int)

    true_flux = np.array([matched[f + "magVar"] for f in filters])

    measurements = []
    for k, (cy, cx) in enumerate(matched_centers):
        # Get the matching index for the source based on its center
        matched_idx = np.where((centers[:, 0] == cy) & (centers[:, 1] == cx))[0][0]

        # Calculate the flux difference in each band
        source = sources[matched_idx]
        flux = 27 - 2.5*np.log10(scarlet.measure.flux(source))

        truth = true_flux[:, k]

        measurement = {
            "x": cx,
            "y": cy,
            "source_id": k,
        }

        for f in range(len(filters)):
            measurement[filters[f]+" truth"] = truth[f]
            measurement[filters[f]+" mag"] = flux[f]

        measurements.append(measurement)

    return measurements


def check_log(data: np.ndarray, ax: plt.axis):
    """Check to see if the data should use a log scale

    :param data: array that is being plotted
    :param ax: The axis that contains the plot
    :return: Whether or not to use a log scale
    """
    _data = np.log10(data)
    ymin, ymax = np.min(_data), np.max(_data)
    # Use a log scale if the range is more than 2 orders of magnitude
    if ymax - ymin > 2:
        ymin = int(np.max([1e-50, ymin - 1]))
        ymax = int(ymax+1)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        ax.yaxis.set_ticks([
            np.log10(x) for p in range(ymin, ymax)
            for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True)
        return True
    return False


class Metric:
    """A metric to be calculated based on a set of deblended sources
    """
    def __init__(
            self,
            name: str,
            units: str,
    ):
        """Initialize the class

        :param name: Name of the metric.
        :param units: Units of the metric.
        :param use_abs: Whether or not this metric is an absolute value
        """
        self.name = name
        self.units = units

    def plot(
            self,
            set_id: str,
            measurements: Dict[str, np.rec.recarray] = None,
            plot_indices: Sequence = None,
            scatter_indices: Sequence = None,
    ) -> plt.Figure:
        """Create a plot using the records for a given set ID.

        :param set_id: ID of the set to analyze
        :param measurements: Dictionary (branch name, measurments)
            of measurements for each branch.
        :param plot_indices: The indices or slice of `measurements`
            to plot. If `plot_indices` is `None` then only the
            10 latest branches are used.
        :param scatter_indices: The indices or slice of `measurements`
            to include in the scatter plot. If `scatter_indices` is `None`
            then only the last two branches are plotted.
        """
        if measurements is None:
            branches = get_branches()
            measurements = {
                branch: np.load(os.path.join(__DATA_PATH__, set_id, get_filename(branch)))["records"]
                for branch in branches
            }
        if plot_indices is None:
            plot_indices = slice(-10, None)
        if scatter_indices is None:
            scatter_indices = slice(-2, None)

        # First display the scatter plots
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        records = {m: measurements[m] for m in list(measurements.keys())[scatter_indices]}
        num_prs = len(records)

        # Check to see if we need to plot a log axis
        islog = False
        for rec, (branch, record) in enumerate(records.items()):
            islog |= check_log(record[self.name], ax[2])

        # Display the scatter plot for each PR
        for rec, (pr, record) in enumerate(records.items()):
            x = np.arange(len(record[self.name]))
            if islog:
                data = np.log10(record[self.name])
            else:
                data = record[self.name]
            ax[2].scatter(x, data, label=pr, s=10 * (num_prs - rec))
        ax[2].legend()
        ax[2].set_xlabel("blend index")

        # Next create the violin and box plots
        records = {m: measurements[m] for m in list(measurements.keys())[plot_indices]}

        for ax_n, plot_type in enumerate(["box", "violin"]):
            # Extract the data
            x = np.arange(len(records))
            data = []
            for s, (pr, record) in enumerate(records.items()):
                data.append(record[self.name])

            # Check if we need a log plot
            islog = check_log(data, ax[ax_n])
            if islog:
                data = [np.log10(d) for d in data]

            if plot_type == "violin":
                # Make the violin plot
                ax[ax_n].violinplot(data, x, showmeans=False, showextrema=False, showmedians=False)

                # Calculate the quartile whiskers
                quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
                whiskers = np.array([
                    adjacent_values(sorted_array, q1, q3)
                    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
                whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
                # Display the whiskers
                ax[ax_n].scatter(x, medians, marker='o', color='white', s=30, zorder=3)
                ax[ax_n].vlines(x, quartile1, quartile3, color='k', linestyle='-', lw=5)
                ax[ax_n].vlines(x, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
            else:
                # Make the box plot
                ax[ax_n].boxplot(data)

        x_labels = tuple(records.keys())
        ax[1].xaxis.set_ticks(np.arange(len(x_labels)))
        ax[0].set_xticklabels(x_labels, size='small', rotation='vertical')
        ax[1].set_xticklabels(x_labels, size='small', rotation='vertical')

        ax[0].set_ylabel(self.units)
        fig.suptitle(self.name, y=.95)
        plt.tight_layout()

        return fig


# All of the metrics that are stored and plotted for regression testing
all_metrics = {
    "init time": Metric("init time", "time (ms)"),
    "runtime": Metric("runtime", "time/source (ms)"),
    "iterations": Metric("iterations", "iterations"),
    "init logL": Metric("init logL", "logL"),
    "logL": Metric("logL", "logL"),
    "g diff": Metric("g diff", "truth-model"),
    "r diff": Metric("r diff", "truth-model"),
    "i diff": Metric("i diff", "truth-model"),
    "z diff": Metric("z diff", "truth-model"),
    "y diff": Metric("y diff", "truth-model"),
}
