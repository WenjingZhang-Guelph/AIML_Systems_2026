"""Plotting for the Marimo notebook. Every function returns a Figure.

Returning rather than calling plt.show() is the one rule that matters here —
Marimo renders whatever a cell returns, and it's also what lets
gallery/render_marimo_mimic_iii.py drive these headlessly to produce the gallery
images. A stray plt.show() breaks both.

`background` / `background_colour` is threaded through everything because the
same plots get used two ways: dark (#191a1c, the Aurora theme) on screen, white
for anything going into a printed figure. Text colour adapts via is_hex_light.
"""
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import marimo as mo

# Dark base → purple → green. Sequential and roughly perceptually ordered, so it
# reads sensibly in greyscale and doesn't imply a midpoint the data doesn't have.
AURORA_GRADIENT = ["#191a1c", "#724ed5", "#4ED595"]

def is_hex_light(hex_colour: str, threshold: float = 0.5):
    """Should text on this background be dark or light?

    Uses the ITU-R BT.709 luminance weights rather than a plain RGB average,
    because the eye is far more sensitive to green than blue — averaging calls
    pure blue "light" and you end up with white text on a blue background.
    """
    hex_colour = hex_colour.lstrip('#')

    r = int(hex_colour[:2], 16) / 255
    g = int(hex_colour[2:4], 16) / 255
    b = int(hex_colour[4:6], 16) / 255

    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return luminance > threshold

def generate_hex_gradient(colours, gradient_length) -> List[str]:
    """Interpolate N evenly-spaced colours through a list of stops.

    Hand-rolled rather than using LinearSegmentedColormap directly because the
    discrete list is useful on its own — assigning one colour per filter or per
    scenario in a bar chart. The colormap is built from this, not the reverse.
    """
    if len(colours) < 2:
        raise ValueError("At least two colors are required.")

    if gradient_length == 1:
        return [colours[0]]

    rgb_colors = np.array([mcolors.to_rgb(colour) for colour in colours])
    segments = len(colours) - 1

    gradient = []
    for i in range(gradient_length):
        position = i / (gradient_length - 1)
        # min() clamps the last point, which would otherwise index one segment
        # past the end when position hits exactly 1.0.
        segment_index = min(int(position * segments), segments - 1)
        segment_position = (position - segment_index / segments) * segments

        start_rgb = rgb_colors[segment_index]
        end_rgb = rgb_colors[segment_index + 1]

        interpolated_rgb = start_rgb + (end_rgb - start_rgb) * segment_position
        gradient.append(mcolors.to_hex(interpolated_rgb))

    return gradient

def surface_plot(dataframe: pd.DataFrame, title: str = '3D Surface', x_title: str = '', y_title: str = '', z_title: str = '', fig_size: Tuple[int, int] = (12, 6), background_colour: str = '#191a1c'):
    """3D surface over the 24×7 grid. Cosmetic, but good for spotting shape.

    A 24×7 surface plotted raw looks like a staircase, so it's resampled 10× on
    both axes and cubic-interpolated. Purely presentational — the interpolated
    values aren't data, and cubic interpolation can overshoot past the real
    min/max, so don't read peaks off this plot. Use the heatmap for anything
    quantitative.
    """
    rows, columns = dataframe.shape

    x, y = np.meshgrid(np.arange(columns), np.arange(rows))
    z = dataframe.values

    x_dense = np.linspace(0, columns - 1, columns * 10)
    y_dense = np.linspace(0, rows - 1, rows * 10)
    x_grid, y_grid = np.meshgrid(x_dense, y_dense)
    z_smooth = griddata(
        np.column_stack((x.ravel(), y.ravel())),
        z.ravel(),
        (x_grid, y_grid),
        method='cubic'
    )

    gradient_length = 30
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "aurora",
        generate_hex_gradient(AURORA_GRADIENT, gradient_length),
        N=gradient_length
    )

    figure = plt.figure(figsize=fig_size, facecolor=background_colour)
    axes = figure.add_subplot(111, projection='3d', facecolor=background_colour)

    surface = axes.plot_surface(x_grid, y_grid, z_smooth, cmap=cmap, edgecolor='none')

    # Ticks are forced back to the original column count — the dense grid has
    # 10× as many points, and without this matplotlib labels interpolated
    # positions that correspond to no actual vital.
    axes.set(xlabel=x_title, ylabel=y_title, zlabel=z_title, title=title)
    axes.set_xticks(np.arange(columns))
    axes.set_xticklabels(dataframe.columns, rotation=45, ha="right")

    # Reaching into _axinfo is the only way to style 3D panes — mplot3d exposes
    # no public API for it, so this may break on a matplotlib upgrade.
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.pane.fill = False
        axis._axinfo["grid"]['color'] = "#3a3b3d"
        axis._axinfo["grid"]['linewidth'] = 0.5
        axis._axinfo["grid"]['linestyle'] = ':'
        axis._axinfo['edgecolor'] = "#3a3b3d"

    colour_bar = figure.colorbar(surface, ax=axes, shrink=0.6, pad=0.1)
    colour_bar.outline.set_visible(False)

    axes.autoscale()

    return figure

def heatmap(dataframe: pd.DataFrame, title: str = "Heatmap", x_title: str = "", y_title: str = "", text: bool = True, fig_size: Tuple[int, int] = (8, 5), background_colour: str = '#191a1c', value_format: str = ".1f"):
    """The honest counterpart to surface_plot — real cells, real values.

    origin='lower' puts hour 0 at the bottom, so time reads upward the way you'd
    expect from a chart rather than matplotlib's default top-down image convention.
    """
    rows, columns = dataframe.shape

    gradient_length = 30
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "aurora",
        generate_hex_gradient(AURORA_GRADIENT, gradient_length),
        N=gradient_length
    )

    figure, axes = plt.subplots(figsize=fig_size, facecolor=background_colour)

    heatmap = axes.imshow(dataframe.values, cmap=cmap, aspect='auto', origin='lower')

    # NaN cells are left blank rather than printing "nan" — an empty cell reads
    # as "no data" at a glance, which is what it means.
    if text:
        for i in range(rows):
            for j in range(columns):
                val = dataframe.values[i, j]
                if not np.isnan(val):
                    axes.text(j, i, f"{val:{value_format}}", ha='center', va='center', color='white', fontsize=8)

    axes.set(xlabel=x_title, ylabel=y_title, title=title)

    column_names = dataframe.columns
    row_names = dataframe.index
    axes.set_xticks(np.arange(columns))
    axes.set_yticks(np.arange(rows))
    axes.set_xticklabels(column_names, rotation=45, ha='right', color='white')
    axes.set_yticklabels(row_names, color='white')

    # Minor ticks offset by half a cell so gridlines land *between* cells rather
    # than through their centres — that's what draws the cell borders.
    axes.set_xticks(np.arange(-0.5, columns, 1), minor=True)
    axes.set_yticks(np.arange(-0.5, rows, 1), minor=True)

    axes.grid(which="minor", color="#3a3b3d", linestyle=':', linewidth=0.5)
    axes.tick_params(which="minor", bottom=False, left=False)

    axes.set_frame_on(False)
    for spine in axes.spines.values():
        spine.set_visible(False)

    colour_bar = figure.colorbar(heatmap, ax=axes, shrink=0.7)
    colour_bar.outline.set_visible(False)

    return figure

def filter_impact_table(results: Tuple[List], filter_names: List[str], delta:bool=False):
    """Tabulate evaluate_filter_impact's five parallel lists.

    `results` is positional and undocumented at the source, so for reference:
    (cv accuracy, test accuracy, train F1, test F1, McNemar (stat, p) pairs).
    Index 0 of each is the raw baseline, which is why `delta` subtracts [0].

    Two things carried over from evaluation_manager that show up here: "training
    accuracy" is really the cross-validation score, and "Training F1" is scored
    on the training split itself, so it sits near 1.0 for a random forest —
    read it as overfit headroom, not model quality.

    The testing columns are scored on a true held-out 10% at a threshold chosen
    on the validation split, not at the implicit 0.5. Expect lower accuracy than
    pre-2026-08-10 tables showed, particularly for mortality — that's the
    threshold trading accuracy for sensitivity on a low-prevalence label, not a
    regression. evaluation_manager's diagnostics sidecar carries the thresholds.

    Watch out — filter_names.insert() mutates the caller's list. Passing the same
    list twice gets you 'Raw', 'Raw', ... so hand it a copy.
    """
    rows = []

    filter_names.insert(0, 'Raw')

    for i in range(len(results[0])):
        if delta:
            rows.append({
                'Filter Name': filter_names[i],
                'Training Accuracy Delta': results[0][i] - results[0][0],
                'Testing Accuracy Delta': results[1][i] - results[1][0],
                'Training F1 Delta': results[2][i] - results[2][0],
                'Testing F1 Delta': results[3][i] - results[3][0],
                'McNemar Statistic Delta': results[4][i][0] - results[4][0][0],
                'McNemar P-Value Delta': results[4][i][1] - results[4][0][1],
            })
        else:
            rows.append({
                'Filter Name': filter_names[i],
                'Training Accuracy': results[0][i],
                'Testing Accuracy': results[1][i],
                'Training F1': results[2][i],
                'Testing F1': results[3][i],
                'McNemar Statistic': results[4][i][0],
                'McNemar P-Value': results[4][i][1],
            })

    return mo.ui.table(rows, selection=None, pagination=False)

def filter_impact_plot(scores: List[float], filter_names: List[str], label: str, metric: str, aggregation_method: str, epsilon: float = 1e-3, min_range: float = 0.005, fig_size: Tuple[int, int] = (14, 8), background:
str='#191a1c'):
    """Bar chart of how far each filter moved the metric off the raw baseline.

    Plots deviations rather than raw scores deliberately. All these accuracies
    sit within a percent or two of each other, so absolute bars all look
    identical; against a zero line the differences are actually visible.

    `epsilon` is the "don't kid yourself" threshold — anything within ±0.001 of
    baseline is greyed out rather than coloured green or red, because at this
    sample size a change that small is seed noise, not a real effect.
    """
    baseline_scores = scores[0]

    filtered_scores = np.array(scores[1:])
    indices = np.arange(len(filtered_scores))

    deviations = filtered_scores - baseline_scores

    bar_colors = np.where(deviations > epsilon, '#3b8465', np.where(deviations < -epsilon, '#bd3140', '#a0a0a0'))

    figure, axes = plt.subplots(figsize=fig_size, facecolor=background)

    axes.bar(indices, deviations, color=bar_colors, width=0.8, zorder=3)

    for i, deviation in enumerate(deviations):
        vertical_alignment = 'bottom' if deviation >= 0 else 'top'
        offset = 0.0005 if deviation >= 0 else -0.0005
        axes.text(i, deviation + offset, f'{deviation:+.4f}', ha='center', va=vertical_alignment, color='#a0a0a0', fontsize=9, fontweight='bold')

    for index in indices[deviations > epsilon]:
        axes.axvline(x=index, color='#3b8465', linestyle='-', linewidth=0.5, alpha=0.3)
    for index in indices[deviations < -epsilon]:
        axes.axvline(x=index, color='#bd3140', linestyle='-', linewidth=0.5, alpha=0.3)

    axes.set_xticks(np.arange(len(filter_names)))
    axes.set_xticklabels(filter_names, rotation=45, ha='right', fontsize=10)

    # y-axis floor. Without it, a run where every filter does essentially nothing
    # gets autoscaled down to a ±0.0001 range and the noise looks like a dramatic
    # effect. Clamping the axis makes "nothing happened" actually look like
    # nothing happened.
    min_deviation, max_deviation = deviations.min(), deviations.max()
    deviation_range = max_deviation - min_deviation

    if deviation_range < min_range:
        axes.set_ylim(-min_range * 1.2, min_range * 1.2)
    else:
        y_margin = deviation_range * 0.25
        axes.set_ylim(min_deviation - y_margin, max_deviation + y_margin)

    axes.patch.set_facecolor(background)
    axes.set_facecolor(background)
    axes.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2, zorder=0)

    foreground = '#000000' if is_hex_light(background) else '#ffffff'

    axes.spines['bottom'].set_color(foreground)
    axes.spines['left'].set_color(foreground)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.tick_params(colors=foreground)
    axes.xaxis.label.set_color(foreground)
    axes.yaxis.label.set_color(foreground)
    axes.title.set_color(foreground)

    axes.set_xlabel('Filter Name')
    axes.set_ylabel(f'{metric} Deviation from Baseline')
    axes.set_title(f'{label} Filter Impact {metric} Deviation From Raw Using {aggregation_method.capitalize()} Aggregation')

    axes.axhline(0, color=foreground, linewidth=0.8, alpha=0.5)

    plt.tight_layout()

    return figure

def mcnemar_plot(p_values, filter_names, label: str, aggregation_method: str, figsize: Tuple[int, int]=(14, 7), background: str='#191a1c'):
    """McNemar p-values as -log10, so taller bars mean more significant.

    Plotted on a log scale because raw p-values bunch up against zero — every
    interesting result would be a bar of indistinguishable height at the bottom.
    After -log10, p=0.05 is 1.3, p=0.001 is 3, and differences are readable.

    The 1e-20 floor exists because an exact McNemar test can return a p-value
    small enough to underflow to 0, and log10(0) is -inf, which crashes the plot.
    Anything hitting the floor is astronomically significant anyway, so the
    clamp costs nothing.

    Colour and the dashed line both mark p=0.05. Worth remembering that's an
    uncorrected threshold and there are 11 filters on this chart — expect roughly
    one false positive per plot by chance alone.
    """
    p_values = [max(p, 1e-20) for p in p_values]

    negative_log_p_value = -np.log10(p_values)

    indices = np.arange(len(p_values))
    # Truncate rather than assume they match — some sweeps have fewer p-values
    # than names, and zip-style misalignment would mislabel every bar.
    filter_names = filter_names[:len(p_values)]

    figure, axis = plt.subplots(figsize=figsize, facecolor=background)
    axis.set_facecolor(background)

    colors = ['#f39c12' if p < 0.05 else '#a0a0a0' for p in p_values]
    bars = axis.bar(indices, negative_log_p_value, color=colors, alpha=0.85, zorder=3)

    threshold = 0.05
    axis.axhline(y=-np.log10(threshold), color='#bd3140', linestyle='--', linewidth=1.5, label=f'p = {threshold} Threshold', zorder=4)

    foreground = '#000000' if is_hex_light(background) else '#ffffff'

    axis.set_xticks(indices)
    axis.set_xticklabels(filter_names, rotation=45, ha='right', color=foreground)
    axis.tick_params(axis='y', colors=foreground)
    axis.set_ylabel(r'$-\log_{10}(p\text{-value})$', color=foreground, fontsize=12)
    axis.set_xlabel('Filter Name', color=foreground, fontsize=12)
    axis.set_title(f'{label.upper()} McNemar', color=foreground, fontsize=16, pad=20)

    # Print the actual p-value on each bar, since a -log10 height is hard to read
    # back. Switches to scientific notation below 0.001, where fixed-point would
    # just show "0.000".
    for bar, p_value in zip(bars, p_values):
        height = bar.get_height()
        display_p = f"{p_value:.3f}" if p_value > 0.001 else f"{p_value:.1e}"
        axis.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                  display_p, ha='center', va='bottom', color='#a0a0a0', fontsize=9)

    axis.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.2, zorder=0)
    for spine in axis.spines.values():
        spine.set_color('#444444')

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    # Second set_title call overrides the one above — the first is dead. Left
    # alone rather than tidied, since the two produce different text and I'd
    # rather not silently change what the plots say.
    axis.title = axis.set_title(f'{label} McNemar Significance Using {aggregation_method.title()} Aggregation', color=foreground)

    axis.legend(facecolor=background, framealpha=0.8, edgecolor='white', labelcolor='white')
    plt.tight_layout()

    return figure

def centroid_shift_plot(centroid_pairs: List[Tuple[List[float], List[float]]], categories: List[str], filter_name: str, vital_names: List[str], epsilon: float = 1e-3, fig_size: Tuple[int, int]=(20, 6), background: str='#191a1c'):
    """Grouped bars: how far a filter moved each vital's centroid, per class.

    Colour encodes direction (up/down/negligible), so it can't also encode which
    of the four groups a bar belongs to — hence hatching. It's ugly, but with
    four groups × seven vitals on one axis it's the only way to tell ICU-positive
    from mortality-negative at a glance, and it survives greyscale printing.

    Same epsilon convention as filter_impact_plot: within ±0.001 is grey, not
    green or red.
    """
    figure, axes = plt.subplots(figsize=fig_size, facecolor=background)

    num_groups = len(centroid_pairs)
    indices = np.arange(len(centroid_pairs[0][0]))
    # 0.8 total width leaves a visible gap between vitals.
    bar_width = 0.8 / num_groups

    hatches = ['', '//', '..', 'xx', '\\\\', '++', 'OO', '--']

    green_base = '#3b8465'
    red_base = '#bd3140'
    gray_base = '#a0a0a0'

    foreground = '#000000' if is_hex_light(background) else '#ffffff'

    for g, ((baseline, scores), title_suffix) in enumerate(zip(centroid_pairs, categories)):
        baseline = np.array(baseline)
        scores = np.array(scores)
        deviations = scores - baseline

        # Shifts each group off the tick so they sit side by side, centred on it.
        pos = indices + (g - num_groups / 2) * bar_width + bar_width / 2

        colors = [green_base if dev > epsilon else red_base if dev < -epsilon else gray_base for dev in deviations]

        current_hatch = hatches[g % len(hatches)]

        bars = axes.bar(pos, deviations, color=colors, width=bar_width, edgecolor=foreground, linewidth=0.5, hatch=current_hatch, label=title_suffix, zorder=3)

        # Exact zero gets no label — an unchanged vital doesn't need "+0.0000"
        # cluttering the axis.
        for i, deviation in enumerate(deviations):
            va = 'bottom' if deviation >= 0 else 'top'
            offset = 0.0005 if deviation >= 0 else -0.0005
            axes.text(pos[i], deviation + offset, '' if deviation == 0 else f'{deviation:+.4f}',
                    ha='center', va=va, color='#a0a0a0', fontsize=8, fontweight='bold')

    axes.set_xticks(indices)
    axes.set_xticklabels([key.title() for key in vital_names], rotation=45, ha='right', fontsize=10)

    axes.patch.set_facecolor(background)
    axes.set_facecolor(background)
    axes.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2, zorder=0)

    for spine in axes.spines.values():
        spine.set_color(foreground)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.tick_params(colors=foreground)
    axes.xaxis.label.set_color(foreground)
    axes.yaxis.label.set_color(foreground)
    axes.title.set_color(foreground)

    axes.set_xlabel('Vital Name')
    axes.set_ylabel('Deviation From Baseline')
    axes.set_title(f'Centroid Deviations ({filter_name.title()} Filter)')
    axes.axhline(0, color=foreground, linewidth=0.8, alpha=0.5)

    # The legend is what makes the hatching decipherable — don't drop it.
    axes.legend(facecolor=background, labelcolor=foreground)

    # tight_layout left off deliberately: with 4 groups × 7 vitals it crops the
    # rotated x-labels.

    return figure

def centroid_plot(centroids: List[List[float]], points: List[List[List[float]]], vitals: List[str], units: List[str], categories: List[str], figsize: Tuple[int, int]=(30, 10), background: str= '#191a1c', use_percentile: bool = True):
    """One violin-ish density panel per vital, overlaying every group.

    The companion to centroid_shift_plot: that one says how far the centroid
    moved, this shows the spread it moved within. A shift that's small next to
    the distribution isn't worth anything, and only this plot tells you that.

    `use_percentile` is the important knob. Clipping each group at the 95th
    percentile of distance-from-centroid keeps a handful of extreme records from
    stretching the axis until the actual distribution is a flat line. Setting it
    False shows the true extent — worth checking, since sometimes those outliers
    are the story. The gallery renders both versions of every panel for exactly
    this reason.
    """
    num_centroids = len(centroids[0])

    # Fixed 2×4 because there are seven vitals; the eighth cell is blanked below.
    plot_columns = 4
    plot_rows = 2

    figure = plt.figure(figsize=figsize, facecolor=background)
    grid_spec = figure.add_gridspec(plot_rows, plot_columns)

    axes = []

    for i in range(num_centroids):
        row = i // plot_columns
        column = i % plot_columns
        axes.append(figure.add_subplot(grid_spec[row, column]))

    if num_centroids < plot_rows * plot_columns:
        for j in range(num_centroids, plot_rows * plot_columns):
            row = j // plot_columns
            column = j % plot_columns
            ax = figure.add_subplot(grid_spec[row, column])
            ax.axis("off")

    colors = ["#34eb9b", "#1e8a5b", "#f2c14e", "#d98f2b", "#8b5a2b", "#5c3a1e", "#ff4d4d", "#b22222", "#a66bff", "#5a2ea6"]

    num_groups = len(centroids)
    # Nudges each group sideways so overlapping densities stay distinguishable.
    offsets = np.linspace(-0.25, 0.25, num_groups)

    for i in range(num_centroids):
        axis = axes[i]

        for g in range(num_groups):
            group_centroids = centroids[g]
            group_points = np.asarray(points[g])
            group_color = colors[g % len(colors)]
            group_offset = offsets[g]

            values = group_points[:, i]
            center = group_centroids[i]

            # Clip by distance from the centroid rather than by absolute value,
            # so it works the same for every vital regardless of scale — no
            # per-vital thresholds to maintain.
            distances = np.abs(values - center)
            limit = np.nanpercentile(distances, 95) if use_percentile else np.nanmax(distances)
            values = values[distances <= limit]

            if values.size == 0:
                continue

            bins = 80
            hist, edges = np.histogram(values, bins=bins, density=True)

            y = (edges[:-1] + edges[1:]) / 2
            density = hist / (hist.max() + 1e-9)

            band = 0.03
            x_left = group_offset - density * band
            x_right = group_offset + density * band

            axis.fill_betweenx(
                y,
                x_left,
                x_right,
                color=group_color,
                alpha=0.45
            )

            axis.plot(group_offset, center,
                      marker='X',
                      color=group_color,
                      markersize=10,
                      label=categories[g] if i == 0 else None)

            axis.set_title(vitals[i])
            axis.set_xticks([])
            axis.set_xlabel('Groups')
            axis.set_ylabel(units[i])
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.patch.set_facecolor(background)
            axis.set_facecolor(background)
            axis.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2, zorder=0)

    figure.legend(handles=axes[0].get_legend_handles_labels()[0],
                  labels=axes[0].get_legend_handles_labels()[1],
                  loc='upper center',
                  ncol=len(categories),
                  bbox_to_anchor=(0.5, 1.02))

    figure.subplots_adjust(wspace=0.25, hspace=0.35)

    return figure