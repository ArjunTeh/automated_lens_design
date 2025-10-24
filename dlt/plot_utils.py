import os
import numpy as np
import jax
import jax.numpy as jnp

import plotly.graph_objects as go
import plotly.colors as pcolors
from plotly.express.colors import sample_colorscale
from plotly.subplots import make_subplots

from dlt import constants
from dlt import zemax
from dlt import zemax_loader
from dlt import tracing
from dlt import curvature_sphere
from dlt import aspheric
from dlt import opt_utils
from dlt import optical_properties
from dlt import system_matrix
from dlt import primary_sample_space as pss

from mcmc import chainstate as cs

from typing import List, Dict

Reservoir = List[cs.ChainState]

def plot_lens(result : pss.NormalizedLens, angles, focal_length=50.0, nrays=20, ignore_L=False, ray_fill=False, ray_width=1.0, lens_width=3.0):
    res_unnorm = pss.normalized2lens(result)
    lens = res_unnorm.data[:result.nsurfaces]

    line_args = dict(color='black', width=lens_width)
    traces = visualize_state_and_rays(lens, angle_list=angles, focal_length=focal_length, nrays=nrays, line_args=line_args, show_invalid_rays=False, ignore_L=ignore_L, ray_fill=ray_fill, ray_width=ray_width)
    return traces


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
    traces = visualize_state_and_rays(lens_data, angle_list=[0.0, jnp.deg2rad(4.0)], focal_length=50.0)
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
        traces = visualize_state(lens_data, offset=offset)
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


def curvature_aa_surface_to_points(k, origin, max_height=10.0):
    max_height = np.minimum(max_height, 1 / np.maximum(np.abs(k), constants.FLOAT_EPSILON))
    y = np.linspace(-max_height, max_height, 201)
    if np.isclose(k, 0):
        x = np.zeros_like(y)
    else:
        x = (1 - np.sqrt(1 - np.minimum(k**2 * y**2, 1.0))) / k

    return x + origin, y


def plot_curvature_sphere(k, origin, max_height=10.0, line_args=None):
    x, y = curvature_aa_surface_to_points(k, origin, max_height)
    return go.Scatter(x=x, y=y, line=line_args)


def visualize_lens_file(fig, fname):
    lens, lens_desc = zemax_loader.load_zemax_file(fname, info_list=True)
    fig.add_traces(visualize_state(lens, lens_desc, offset=-lens[:, 1].sum()))
    return fig


def visualize_state(lens, lens_desc=None, line_args=None, colorscale='blues', offset=0.0):
    if lens_desc is None:
        lens_desc = [zemax_loader.ZemaxSurfaceType.STANDARD] * lens.shape[0]

    if line_args == None:
        line_args = dict(color='black', width=2.0)

    color_list = pcolors.sample_colorscale(colorscale, samplepoints=list((np.array(lens[:, 2]).clip(min=1.0) - 1) / 2.0))

    lens = optical_properties.clip_semidiameter_to_curvature(lens)

    traces = []
    for i in range(lens.shape[0]):
        if lens_desc[i] == zemax_loader.ZemaxSurfaceType.STOP:
            trace = plot_stop(origin=offset, height=lens[i, 3], thickness=0.1, max_rad=1.5*lens[i, 3], line_args=line_args)
            traces.append(trace) 
        elif lens[i, 2] > (lens[-1, 2] + 1e-5):
            # draw the lens
            trace = plot_asphere_lens(lens[i],
                                      lens[i+1], offset,
                                      max_height=lens[i, 3], max_height2=lens[i+1, 3], 
                                      line_args=line_args, fillcolor=color_list[i])
            traces.append(trace) 
        offset += lens[i, 1]

    return traces


def visualize_state_and_rays(lens, lens_desc=None, line_args=None, colorscale='blues', focal_length=1.0, nrays=20, beam_size=1.0, angle_list=[0.0], object_distance=np.inf, show_invalid_rays=True, ignore_L=False, ray_fill=False, ray_width=1.0):
    offset = -lens[:, 1].sum()
    traces = []
    traces += visualize_state(lens, lens_desc, line_args, colorscale, offset)
    traces += plot_rays(lens, focal_length, nrays, beam_size, angle_list, object_distance, show_invalid_rays, ignore_L, ray_fill=ray_fill, line_width=ray_width)
    return traces


def plot_sphere_lens(k1, o1, k2, o2, max_height=10.0, max_height2=None, line_args=None, fillcolor=None):
    if max_height2 is None:
        max_height2 = max_height
    x1, y1 = curvature_aa_surface_to_points(k1, o1, max_height)
    x2, y2 = curvature_aa_surface_to_points(k2, o2, max_height2)

    x = np.concatenate([x1, x2[::-1]])
    y = np.concatenate([y1, y2[::-1]])
    x = np.concatenate([x, [x[0]]])
    y = np.concatenate([y, [y[0]]])

    return go.Scatter(x=x, y=y, line=line_args, fill='toself', fillcolor=fillcolor)


def plot_asphere_lens(lens1, lens2, offset, max_height=10.0, max_height2=None, line_args=None, fillcolor=None):
    if max_height2 is None:
        max_height2 = max_height

    ho1 = lens1[4:] if lens1.shape[0] > 4 else np.zeros(8)
    ho2 = lens2[4:] if lens2.shape[0] > 4 else np.zeros(8)

    x1, y1 = asphere_aa_surface_to_points(lens1[0], offset, ho1, max_height)
    x2, y2 = asphere_aa_surface_to_points(lens2[0], offset + lens1[1], ho2, max_height2)

    x = np.concatenate([x1, x2[::-1]])
    y = np.concatenate([y1, y2[::-1]])
    if not x.size == 0:
        x = np.concatenate([x, [x[0]]])
        y = np.concatenate([y, [y[0]]])

    return go.Scatter(x=x, y=y, line=line_args, fill='toself', fillcolor=fillcolor, opacity=1.0)


def asphere_aa_surface_to_points(k, origin, higher_order, max_height=10.0):
    if not np.isfinite(k) or not np.all(np.isfinite(higher_order)):
        return [origin], [0]

    conic = higher_order[0]

    y = np.linspace(-max_height, max_height, 200)
    y_det = 1 - (1 + conic) * k**2 * y**2
    valid = y_det >= 0
    y_det_safe = np.where(valid, y_det, 0.0)

    x = k * y**2 / (1 + np.sqrt(y_det_safe))

    for i, ho in enumerate(higher_order[1:]):
        x += ho * y**(2*i+2)

    return x[valid] + origin, y[valid]


def plot_stop(origin, height, thickness=0.1, max_rad=None, line_args=None):
    o = origin
    h = height
    th = thickness
    mr = max_rad if max_rad is not None else 1.5 * th

    x = [o, o, None, o, o, None, o-th, o+th, None, o-th, o+th]
    y = [h, mr, None, -h, -mr, None, h, h, None, -h, -h]

    return go.Scatter(x=x, y=y, mode='lines', line=line_args)


def plot_rays(lens, focal_length, nrays=20, beam_size=1.0, angle_list=[0.0], object_distance=np.inf, show_invalid_rays=True, ignore_L=False, title='', ray_fill=False, line_width=1.0):
    '''Plot the state of the lens, the rays, and the targets
        DOES NOT REPARAMETERIZE ASPHERES, do it before calling this function
    '''
    lens = optical_properties.clip_semidiameter_to_curvature(lens)
    sensor_pos = lens[:, 1].sum()
    line_range = lens[0, 3] * beam_size
    if lens.shape[1] > 4:
        surf_funs = [aspheric.functionSuite(asphere_type='standard')] * (lens.shape[0])
        surf_funs.append(curvature_sphere.functionSuite())
    else:
        surf_funs = [curvature_sphere.functionSuite()] * (lens.shape[0] + 1)
    surfs = zemax.zemax_state_to_jax_surfaces(lens, scene_scale=1.0, focal_length=focal_length, object_distance=object_distance, reparameterize=False)
    surfs = jax.numpy.stack(surfs)

    x, v, L, targets = [], [], [], []
    for angle in angle_list:
        cx, cv, aper_rad = optical_properties.chief_ray_by_trace(lens, angle, 50)
        offsets = np.linspace(-aper_rad, aper_rad, nrays)
        xa = np.tile(cx, (offsets.shape[0], 1))
        xa[:, 1] += offsets
        va = np.tile(cv, (offsets.shape[0], 1))
        La = np.ones(xa.shape[0])
        xa = xa - 10 * va
        x.append(xa)
        v.append(va)
        L.append(La)
        target = np.tile(np.array([lens[:, 1].sum(), -np.tan(angle) * focal_length, 0.0]), (xa.shape[0], 1))
        targets.append(target)
    x = np.concatenate(x, axis=0)
    v = np.concatenate(v, axis=0)
    L = np.concatenate(L, axis=0)
    targets = np.concatenate(targets, axis=0)

    vtracer = jax.vmap(tracing.trace_scan_no_stop, (None, None, 0, 0, 0, None, None))
    vtrace_jit = jax.jit(lambda *args: vtracer(surf_funs[0], *args))
    (valid, xp, vp, Lp), warpfields = vtrace_jit(surfs, x, v, L, 0, lens.shape[0])
    xp = xp.swapaxes(0, 1)
    vp = vp.swapaxes(0, 1)
    Lp = Lp.swapaxes(0, 1)

    if focal_length < 0:
        targets = np.zeros_like(targets)
        for i in range(len(angle_list)):
            ray_idx = slice(i*nrays // len(angle_list), (i+1)*nrays // len(angle_list))
            targets[ray_idx] = np.where(valid[ray_idx, None], xp[-1][ray_idx], 0.0).sum(axis=0, keepdims=True) / np.count_nonzero(valid)

    xpath = np.array(xp)
    vpath = np.array(vp)
    Lpath = Lp[-1]

    if ignore_L:
        Lpath = np.ones_like(Lpath)


    yes_colors = ['black']
    no_colors = ['red', 'salmon', 'tomato']

    rays_traces = []
    rays_traces_no = []
    circle_trace = []
    for i, a in enumerate(angle_list):
        step = xpath.shape[1] // len(angle_list)
        step_range = slice(i*step, (i+1)*step)
        xp_view = xpath[:, step_range]
        xp_yes = xp_view[:, valid[step_range]]
        xp_no = xp_view[:, ~valid[step_range]]
        L_yes = Lpath[step_range][valid[step_range]]
        L_no = Lpath[step_range][~valid[step_range]]
        xp_yes[:, :, 0] -= sensor_pos
        xp_no[:, :, 0] -= sensor_pos
        if ray_fill:
            xp_view[:, :, 0] -= sensor_pos
            xp_yes = np.swapaxes(xp_yes, 0, 1)
            xp_no = np.swapaxes(xp_no, 0, 1)
            rays_traces += plot_rays_in_fan(xp_yes, None, line_args=dict(color=yes_colors[i%len(yes_colors)]), title=title)
            rays_traces_no += plot_rays_in_fan(xp_no, None, line_args=dict(color=no_colors[i%len(no_colors)]), title=title)
        else:
            rays_traces += plot_rays_plotly(xp_yes, vpath, L=L_yes, line_args=dict(width=line_width, color=yes_colors[i%len(yes_colors)]))
            rays_traces_no += plot_rays_plotly(xp_no, vpath, L=0.5*np.ones_like(L_no), line_args=dict(width=line_width, color=no_colors[i%len(no_colors)]))

    rays_traces = rays_traces + rays_traces_no if show_invalid_rays else rays_traces

    return rays_traces + circle_trace


def plot_rays_with_tracer(tracer_fn, lens, nsurfaces, offset=0.0, show_invalid_rays=True):
    valid, xp, Lp = tracer_fn(lens, nsurfaces)
    xp = xp.swapaxes(0, 1)
    Lp = Lp.swapaxes(0, 1)

    Lpath = Lp[nsurfaces]
    xpath = np.array(xp)

    yes_colors = ['blue', 'slateblue', 'skyblue']
    no_colors = ['red', 'salmon', 'tomato']

    x = xpath[:nsurfaces+2, valid]
    x[:, :, 0] += offset
    L = Lpath[valid]

    rays_traces = plot_rays_plotly(x, None, L=L, line_args=dict(color=yes_colors[0]))

    if show_invalid_rays:
        xnot = xpath[:nsurfaces+2, ~valid]
        xnot[:, :, 0] += offset
        Lnot = Lpath[~valid]
        rays_traces += plot_rays_plotly(xnot, None, L=0.5*np.ones_like(Lnot), line_args=dict(color=no_colors[0]))

    return rays_traces



def plot_state(lens, focal_length, nrays=20, beam_size=1.0, angle_list=[0.0], object_distance=np.inf, show_invalid_rays=True, title=''):
    rays_traces = plot_rays(lens, focal_length, nrays, beam_size, angle_list, object_distance, show_invalid_rays, title)
    lens_traces = plot_zemax_state(lens, line_args=dict(color='fuchsia'), sensor_at_origin=True)
    return lens_traces + rays_traces

    
def plot_zemax_state(lens_state, line_args=None, sensor_at_origin=False):
    if line_args is not None:
        legendgroup = line_args["color"] + "lenses"
    else:
        legendgroup = "lenses"
    traces = []
    sensor_pos = lens_state[:, 1].sum()
    dist = 0
    for i in range(lens_state.shape[0]):
        k = lens_state[i, 0]
        o = dist 
        ho_terms = lens_state[i, 4:]
        height = float(lens_state[i, 3]) + 0.01
        if lens_state.shape[1] > 4:
            x, y = asphere_aa_surface_to_points(k, o, ho_terms, height)
        else:
            x, y = curvature_aa_surface_to_points(k, o, height)
        x = x-sensor_pos if sensor_at_origin else x
        traces.append(go.Scatter(x=x, y=y, line=line_args, legendgroup=legendgroup))
        dist += lens_state[i, 1]
    return traces


def plot_rays_plotly(xp, vp, L=None, line_args=None, title='rays'):
    if xp.size == 0:
        return []
    if line_args is None:
        line_args = dict(color='blue')
    if L is None:
        L = np.ones((xp.shape[1],))
    rad_max = L.max()
    traces = []
    for i in range(xp.shape[1]):
        trace = go.Scatter(
            x=xp[:, i, 0],
            y=xp[:, i, 1],
            mode="lines",
            line=line_args,
            opacity=1.0 * float(L[i] / rad_max),
            legendgroup=title + line_args["color"]
        )
        traces.append(trace)
    return traces

def plot_rays_in_fan(xp, valid=None, line_args=None, title='rays'):
    if valid is None:
        valid = np.ones((xp.shape[0],), dtype=bool)

    if xp.size == 0:
        return []

    hivalid_idx = -1
    lovalid_idx = 0
    for i in range(xp.shape[0]-1):
        if valid[i] and not valid[i+1]:
            hivalid_idx = i
        if not valid[i] and valid[i+1]:
            lovalid_idx = i + 1

    xvals = np.concatenate([xp[hivalid_idx, :, 0], xp[lovalid_idx, ::-1, 0], xp[hivalid_idx, 0:1, 0]])
    yvals = np.concatenate([xp[hivalid_idx, :, 1], xp[lovalid_idx, ::-1, 1], xp[hivalid_idx, 0:1, 1]])
    return [go.Scatter(x=xvals, y=yvals, fill='toself', line=line_args, opacity=0.3)]


def color_to_rgba(color_name, opacity=1.0):
    pcolors.label_rgb(color_name)
    rgb = pcolors.hex_to_rgb(pcolors.label_rgb(color_name))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'
