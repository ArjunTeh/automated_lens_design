import context

import os

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

import configargparse

from jaxtyping import Key, Float
from typing import Callable, Tuple, Dict, Union, List

import time
from tqdm import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dlt import opt_utils
from dlt import optical_properties
from dlt import primary_sample_space as pss
from dlt import zemax_loader
from dlt import plot_utils
from dlt import mcmc

Reservoir = List[mcmc.ChainState]

def plot_lens(result : pss.NormalizedLens, angles, focal_length=50.0, nrays=20, ignore_L=False, ray_fill=False, ray_width=1.0, lens_width=3.0):
    res_unnorm = pss.normalized2lens(result)
    lens = res_unnorm.data[:result.nsurfaces]

    line_args = dict(color='black', width=lens_width)
    traces = plot_utils.visualize_state_and_rays(lens, angle_list=angles, focal_length=focal_length, nrays=nrays, line_args=line_args, show_invalid_rays=False, ignore_L=ignore_L, ray_fill=ray_fill, ray_width=ray_width)
    return traces


def quick_plot_lens(lens : pss.NormalizedLens, focal_length, nrays=20):
    angles = jnp.atan(jnp.linspace(0,  21, 2) / focal_length)
    fig = go.Figure()
    traces = plot_lens(lens, angles, focal_length=focal_length, nrays=nrays, ignore_L=True)
    fig.add_traces(traces)
    fig.show()


def export_lenses(outfolder, results : Dict[str, pss.NormalizedLens], fig_dimensions, fig_scale, angles, focal_length=50.0, nrays=20, ignore_L=False, ray_fill=False, ray_width=1.0, add_sensor=False):
    max_semidiam = 0
    max_track_length = 0
    for name, result in results.items():
        res_unnorm = pss.normalized2lens(result)
        lens = res_unnorm.toarray()
        max_semidiam = max(max_semidiam, lens[:, 3].max().item())
        max_track_length = max(max_track_length, lens[:, 1].sum().item())

    fig = go.Figure()
    for name, result in results.items():
        fig.data = []
        traces = plot_lens(result, angles, focal_length=focal_length, nrays=nrays, ignore_L=ignore_L, ray_fill=ray_fill, ray_width=ray_width)
        fig.add_traces(traces)
        fig.update_xaxes(range=[-max_track_length-2, 3], visible=False)
        fig.update_yaxes(range=[-1.05*max_semidiam, max_semidiam*1.05], visible=False)
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            # fillcolor="black",
            line=dict(
                width=10,  # Adjust border width as needed
                color="black"
            )
        )
        if add_sensor:
            fig.add_shape(
                type="line",
                x0=0, y0=-max_semidiam, x1=0, y1=max_semidiam,
                line=dict(width=4, color="black")
            )
        outname = os.path.join(outfolder, f"{name}.svg")
        fig.write_image(outname, width=fig_dimensions[0]*10, height=fig_dimensions[1]*10, scale=0.1*fig_scale)  # Save the figure as an image


def plot_focus_curves(result : pss.NormalizedLens, angles, name='', color=None):
    res_unnorm = pss.normalized2lens(result)
    lens = res_unnorm.toarray()

    traces = []
    for ang in angles:
        tan, sag, chief = optical_properties.focus_curves(lens, ang, 100, 100)
        traces.extend([
            go.Scatter(y=tan[0], x=tan[1], name=name + f" tan {jnp.rad2deg(ang):.1f}"),
            go.Scatter(y=sag[0], x=sag[1], name=name + f" sag {jnp.rad2deg(ang):.1f}"),
        ])
    return traces


def compare_focus_curves_plot(lenses : list[pss.NormalizedLens], angles, names):
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'coral', 'cyan', 'magenta']
    subplot_titles = [f"Tan {jnp.rad2deg(ang):.1f}" for ang in angles] + [f"Sag {jnp.rad2deg(ang):.1f}" for ang in angles]

    fig = make_subplots(rows=2, cols=len(angles), subplot_titles=subplot_titles)
    for i, ang in enumerate(angles):
        for j, lens in enumerate(lenses):
            color = colors[j % len(colors)]
            lens_reg = pss.normalized2lens(lens)
            tan, sag, chief = optical_properties.focus_curves(lens_reg.toarray(), ang, 100, 100)
            fig.add_trace(go.Scatter(y=tan[0], x=tan[1], name=names[j] + " tan", line=dict(color=color), legendgroup=names[j]), row=1, col=i+1)
            fig.add_trace(go.Scatter(y=sag[0], x=sag[1], name=names[j] + " sag", line=dict(color=color), legendgroup=names[j]), row=2, col=i+1)
    return fig


def plot_field_sweep(lenses : list[pss.NormalizedLens], angles, nres, names, xlabels=None):
    if xlabels is None:
        xlabels = angles
    rms_traces = []
    thrus_traces = []
    data = []
    for i, lens in enumerate(lenses):
        lens_reg = pss.normalized2lens(lens)
        rms_spots, thrus = optical_properties.rms_spot_size(lens_reg.toarray(), angles, nres)
        rms_traces.append(go.Scatter(y=rms_spots, x=xlabels, name=names[i]))
        thrus_traces.append(go.Scatter(y=thrus, x=xlabels, name=names[i]))
        data.append((rms_spots, thrus))

    return rms_traces, thrus_traces, data


def siggraph_figure_fontspec():
    return dict(
        size=8,
        family="Linux Biolinum O",
        color='black'
    )


def siggraph_axis_desc():
    return dict(
        title_font=siggraph_figure_fontspec(),
        tickfont=siggraph_figure_fontspec(),
        showline=True,
        linewidth=2,
        linecolor='black',
        tickwidth=2,
        tickcolor='black',
    )


def format_figure(fig : go.Figure):

    font_spec = siggraph_figure_fontspec()

    fig.update_layout(
        title_font=font_spec,
        legend_font=font_spec,
        legend_title_font=font_spec,
        xaxis_title_font=font_spec,
        xaxis_nticks=2,
        xaxis_showline=True,
        xaxis_linewidth=2,
        xaxis_linecolor='black',
        xaxis_tickfont=font_spec,
        yaxis_title_font=font_spec,
        yaxis_tickfont=font_spec,
        yaxis_nticks=2,
        yaxis_showline=True,
        yaxis_linewidth=2,
        yaxis_linecolor='black',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        legend=dict(
            # orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=font_spec,
        ),
    )


def plot_distortion(result : pss.NormalizedLens, name=''):
    res_unnorm = pss.normalized2lens(result)
    lens = res_unnorm.toarray()

    distortion, heights = optical_properties.distortion_curve(lens, focal_length=50.0, sensor_height=12.0, nsamples=100)
    traces = [
        go.Scatter(y=jnp.abs(distortion), x=heights, name=name),
    ]
    return traces


def plot_lens_and_reservoir(lens : pss.NormalizedLens, reservoir : Reservoir, max_track_length, max_semidiam, save_path=None):
    FIG_PADDING = 5.0
    LENS_PER_ROW = 5
    nres = len(reservoir)
    nrows = 1 + int(jnp.ceil(nres / LENS_PER_ROW).item())
    figspec = [[{"colspan": LENS_PER_ROW} if i == 0 else None for i in range(LENS_PER_ROW)]]
    for i in range(nrows-1):
        figspec.append([{} for _ in range(LENS_PER_ROW)])

    fig = make_subplots(rows=nrows, cols=LENS_PER_ROW,
                        specs=figspec,
                        subplot_titles=(['Lens'] + [f'Reservoir {i}' for i in range(nres)]))
    
    # plot main lens
    lens_unnorm = pss.normalized2lens(lens)
    lens_data = lens_unnorm.toarray()
    traces = plot_utils.visualize_state_and_rays(lens_data, angle_list=[0.0, jnp.deg2rad(4.0)], focal_length=50.0)
    fig.add_traces(traces, rows=1, cols=1)
    fig.update_xaxes(range=[-(max_track_length + FIG_PADDING), FIG_PADDING], row=1, col=1)
    fig.update_yaxes(range=[-(max_semidiam + FIG_PADDING), max_semidiam + FIG_PADDING], row=1, col=1)

    # plot reservoirs
    for i, res in enumerate(reservoir):
        row = i // LENS_PER_ROW + 2
        col = i % LENS_PER_ROW + 1
        rlens = res.cur_state
        lens_unnorm = pss.normalized2lens(rlens)
        lens_data = lens_unnorm.toarray()
        offset = -lens_data[:, 1].sum().item()
        traces = plot_utils.visualize_state(lens_data, offset=offset)
        fig.add_traces(traces, rows=row, cols=col)
        fig.update_xaxes(range=[-(max_track_length + FIG_PADDING), FIG_PADDING], row=row, col=col)
        fig.update_yaxes(range=[-(max_semidiam + FIG_PADDING), max_semidiam + FIG_PADDING], row=row, col=col)

    if save_path is None:
        fig.show()
    else:
        fig.write_image(save_path, width=2400, height=1600)


def plot_loss_percentages(aux_hist):
    losses = {
        'spot': jnp.array([aux['scaled_spot'] for aux in aux_hist]),
        'thru': jnp.array([aux['scaled_thru'] for aux in aux_hist]),
        'thick': jnp.array([aux['scaled_thick'] for aux in aux_hist]),
        'volume': jnp.array([aux['scaled_volume'] for aux in aux_hist]),
        'focal': jnp.array([aux['scaled_focal'] for aux in aux_hist]),
        'nelements': jnp.array([aux['scaled_nelements'] for aux in aux_hist]),
        'track': jnp.array([aux['scaled_track'] for aux in aux_hist]),
    }

    traces = []
    for name, loss in losses.items():
        traces.append(go.Scatter(y=loss, name=name, mode='lines', stackgroup='one'))
    return traces
