import numpy as np
import matplotlib.pyplot as plt
import pathlib
import sys

script_dir = pathlib.Path().parent.absolute()
parent_dir = script_dir.parents[0]
sys.path.append(str(parent_dir))


x_topo, topo = np.load(parent_dir / 'demo data/corrected_topography.npy')

# load colorbar ------------------------------------------------------------------------------------------------------------

from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, Normalize


def plot_abs_geo_v(ds, title_string, fig_string, saving=False):
    """
    Plot absolute geostrophic velocity from an xarray.Dataset.
    
    Parameters:
        ds (xarray.Dataset): Dataset containing all required variables.
        title_string (str): Title for the plot.
        fig_string (str): String for saving the figure.
        saving (bool): Whether to save the figure.
    """

    # Try to load custom colormap, else use default
    try:
        sampled_colors = np.load(parent_dir / 'data/colorbars/colorbar_pink_yellow_blue_green.npy')
        vel_cmap = LinearSegmentedColormap.from_list('custom_cmap', sampled_colors, N=256)
    except FileNotFoundError:
        print("Custom colorbar not found, using default colormap.")
        vel_cmap = plt.get_cmap('RdYlBu_r')

    value_range = [-0.3, -2.00000000e-01, -1.75000000e-01, -1.50000000e-01,
           -1.25000000e-01, -1.00000000e-01, -7.50000000e-02, -5.00000000e-02,
           -2.50000000e-02, 0.0, 2.50000000e-02,  5.00000000e-02,
            7.50000000e-02,  1.00000000e-01,  1.25000000e-01,  1.50000000e-01,
            1.75000000e-01,  2.00000000e-01, 0.3]
    norm = BoundaryNorm(value_range, ncolors=vel_cmap.N, clip=True)
    vticks_vel = [-0.2, -0.1, 0, 0.1, 0.2]

    # Extract variables from dataset
    v = ds['v'].values
    sigma0half = ds['sigma0half'].values
    xhalf = ds['xhalf'].values
    z = ds['z'].values

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    pc = ax.pcolormesh(xhalf, z , v, cmap=vel_cmap, norm=norm, zorder=0)
    con = ax.contour(xhalf, z , v, levels=[-0.2, -0.1, 0, 0.1, 0.2, 0.3], colors='w', linewidths=1.5)
    contour_labels = plt.clabel(con, inline=False, fontsize=10, fmt='%1.1f')
    for label in contour_labels:
        label.set_color('k')  # Set the color of the labels
        label.set_rotation(270)
        label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'))

    #density contours
    ax.contour(xhalf, z , sigma0half, levels=np.arange(27.6, 27.82, 0.02), colors='k', linewidths=0.5)
    ax.contour(xhalf, z , sigma0half, [27.8])
    ax.set_title(title_string, fontsize=12)
    
    ax.plot(x_topo, topo * -1, color=[.3, .3, .3], linewidth=0.5, zorder=3)
    ax.fill_betweenx(topo * -1, x_topo, where=(x_topo <= 205.622405357364), color=[.8, .8, .8], zorder=2)
    right_mask = (x_topo >= 855.0984735257368)
    ax.fill_betweenx(topo[right_mask] * -1, x_topo[right_mask], x2=1000, color=[.8, .8, .8], zorder=2)
    
    ax.set_xlim([155, 925])
    ax.set_ylim([-2000, 0])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=vel_cmap), ax=ax, ticks=vticks_vel, extend='both')
    cbar.set_label('velocity m/s')
    ax.set_xlabel('distance km')
    ax.set_ylabel('depth m')
    #ax.axvline(x=205, color='y', linestyle=':', linewidth=2)
    #ax.axvline(x=856, color='y', linestyle=':', linewidth=2)
    plt.tight_layout()
    if saving:
        plt.savefig(parent_dir / f'figures/abs_geo_v_{fig_string}.png')
    plt.show()


def plot_overturning_transport(
    main_ds, 
    main_label="Main", 
    main_color="tab:blue",
    extra_lines=None,  # List of dicts: [{'ds': Ds2, 'label': 'Other', 'color': 'red'}, ...]
    title="Overturning Transport",
    savepath=None,
    x_limits=None
):
    """
    Plot overturning transport from one or more datasets.

    Parameters:
        main_ds: xarray.Dataset or dict-like, must have 'strf_z', 'piecewise_trapz_z' and 'z'
        main_label: label for main line
        main_color: color for main line
        extra_lines: list of dicts, each with keys 'ds', 'label', 'color'
        savepath: if given, save the figure to this path
    """
    fig, ax = plt.subplots(figsize=(5, 6))
    # Main line
    ax.plot(main_ds['strf_z'], main_ds['z'], color=main_color, label=main_label)
    ax.plot(main_ds['piecewise_trapz_z'], main_ds['z0'], color=main_color, alpha=0.8, linestyle='--') 
    # Extra lines
    if extra_lines is not None:
        for line in extra_lines:
            ds = line['ds']
            label = line.get('label', 'Extra')
            color = line.get('color', None)
            ax.plot(ds['strf_z'], ds['z'], label=label, color=color)
            ax.plot(ds['piecewise_trapz_z'], ds['z0'], color=color, alpha=0.8, linestyle='--')

    # plot the legend outide the plot at the bottom
    ax.legend(loc='upper center', bbox_to_anchor=(0.25, -0.15), ncol=2, frameon=True, fontsize=10)        
    ax.set_ylabel("Depth m")
    ax.set_xlabel("Transport Sv")
    ax.axvline(0, color='k', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title)
    if x_limits is not None:
        ax.set_xlim(x_limits)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()

def plot_horizontal_transport(
    main_ds, 
    main_label="Transport", 
    main_color="tab:blue",
    extra_lines=None,  # List of dicts: [{'ds': Ds2, 'label': 'Other', 'color': 'red'}, ...]
    title="Horizontal Transport",
    savepath=None
):
    """
    Plot horizontal transport from one or more datasets.

    Parameters:
        main_ds: xarray.Dataset or dict-like, must have 'strf_x', 'piecewise_trapz_x' and 'x'
        main_label: label for main line
        main_color: color for main line
        extra_lines: list of dicts, each with keys 'ds', 'label', 'color'
        savepath: if given, save the figure to this path
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    # Main line
    ax.plot(main_ds['xhalf'], main_ds['strf_x'], color=main_color, label=main_label, linewidth=2)
    ax.plot(main_ds['x0'], main_ds['piecewise_trapz_x'], color=main_color, alpha=0.8, linestyle='--') 
    # Extra lines
    if extra_lines is not None:
        for line in extra_lines:
            ds = line['ds']
            label = line.get('label', 'Extra')
            color = line.get('color', None)
            ax.plot(ds['xhalf'], ds['strf_x'], label=label, color=color)
            ax.plot(ds['x0'], ds['piecewise_trapz_x'], color=color, alpha=0.8, linestyle='--')
    ax.legend()
    ax.set_xlabel("Distance km")
    ax.set_ylabel("Transport Sv")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()


def plot_density_space_transport(
    main_Q, main_Q_cum, scaled_depth, 
    main_label="transport", main_color="mediumorchid",
    extra_lines=None,  # list of dicts: {'Q': ..., 'Q_cum': ..., 'label': ..., 'cum_label': ..., 'color': ...}
    densities=None, sigma_bins=None, mean_sigma0=None, z=None,
    title="Transport in Density Space", savepath=None
):
    """
    Plot transport in density space with optional extra lines.

    Parameters:
        main_Q: array, transport (e.g. Q0)
        main_Q_cum: array, cumulative transport (e.g. Q)
        scaled_depth: array, y-axis (scaled depth)
        main_label: label for main_Q
        main_cum_label: label for main_Q_cum
        main_color: color for main lines
        extra_lines: list of dicts, each with keys 'Q', 'Q_cum', 'label',  'color'
        densities: list of density values for secondary y-axis ticks (optional)
        sigma_bins: array of sigma bins (required if densities is given)
        mean_sigma0: mean sigma0 profile (required if densities is given)
        z: depth array (required if densities is given)
        title: title for the plot
        savepath: if given, save the figure to this path
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    ax.plot(main_Q, scaled_depth, linestyle='--', color=main_color, alpha=0.4)
    ax.plot(main_Q_cum, scaled_depth, color=main_color, label=main_label)

    if extra_lines is not None:
        for line in extra_lines:
            ax.plot(line['Q'], scaled_depth, linestyle='--', color=line['color'], alpha=0.4)
            ax.plot(line['Q_cum'], scaled_depth, color=line['color'], label=line['label'])

    ax.legend()

    # Add secondary y-axis for densities if info is provided
    if densities is not None and sigma_bins is not None and mean_sigma0 is not None and z is not None:
        ints = [np.argmin(np.abs(sigma_bins - d)) for d in densities]
        z_ticks = scaled_depth[ints]
        z_ticks = np.insert(z_ticks, 0, 0)
        dens_ticks = ['']
        dens_ticks.extend([f'{density}' for density in densities])
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(z_ticks)
        ax2.set_yticklabels(dens_ticks)
        ax2.set_ylabel(r'Potential density [$kg m^{-3}$]')
        ax2.spines['top'].set_visible(False)

    ax.set_xlabel("Transport [Sv]")
    ax.set_ylabel("Scaled Depth [m]")
    ax.set_title(title)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()

