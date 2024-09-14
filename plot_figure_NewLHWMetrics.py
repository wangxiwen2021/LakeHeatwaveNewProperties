'''
The script used for plotting figures in "More rapid lake heatwave development"
Created by Xiwen Wang in Nanjing University, Jul 30, 2024
xiwen_wang@smail.nju.edu.cn
'''
# %%
import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
import matplotlib
import seaborn as sns
import pymannkendall as mk
import pickle
from matplotlib.lines import Line2D
import cmasher as cmr 
import gzip

matplotlib.rcParams['font.size'] = 7
matplotlib.rcParams['figure.dpi'] = 600
# journal format
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['mathtext.fontset'] = "custom"
matplotlib.rcParams['mathtext.it'] = "Arial:italic"
matplotlib.rcParams['mathtext.rm'] = "Arial"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.size'] = 7
matplotlib.rcParams['axes.linewidth'] = 0.7
matplotlib.rcParams['xtick.major.width'] = 0.7 
matplotlib.rcParams['ytick.major.width'] = 0.7 
matplotlib.rcParams['xtick.major.size'] = 3 
matplotlib.rcParams['ytick.major.size'] = 3 

# %% [markdown]
# # Functions

# %%
# add_equal_axes
# This function is modified from https://bbs.06climate.com/forum.php?mod=viewthread&tid=101847
import matplotlib.transforms as mtransforms
def add_equal_axes(ax, loc, pad=None, h_pad=None, v_pad=None, h_width=None, h_height=None, v_width=None, v_height=None,
                   ha='left', va='lower'):

    '''
    ax : Axes or array_like of Axes
    loc : {'left', 'right', 'bottom', 'top', 'lowerleft', 'upperleft', 'lowerright', 'upperright'}, 
            the location of new ax relative to old ax(es)
    pad : float, space between old and new axes
    width/height: float
        loc='left' or 'right', h_width is the width of new ax, h_height is the height of new ax (default to be the same as old ax)
        loc='bottom' or 'top', v_height is the height of new ax, v_width is the width of new ax (default to be the same as old ax
    ha: horizontal alignment, [left, right], only worked for loc=top/bottom
    va: vertical alignment, [lower, upper], only worked for loc=left/right
    '''

    # Get the size and location of input ax(es)
    axes = np.atleast_1d(ax).ravel()
    bbox = mtransforms.Bbox.union([ax.get_position() for ax in axes])

    # Height and width of original ax
    width = bbox.x1 - bbox.x0
    height = bbox.y1 - bbox.y0

    # New axes is equal to original ax
    if h_width is None:
        h_width = width
    if v_width is None:
        v_width = width
    if h_height is None:
        h_height = height
    if v_height is None:
        v_height = height

    if h_pad is None:
        h_pad = pad 
    if v_pad is None:
        v_pad = pad 

    # Determine the location and size of new ax
    if loc == 'left':
        x0_new = bbox.x0 - pad - h_width
        x1_new = x0_new + h_width
        if va == 'lower':
            y0_new = bbox.y0 
            y1_new = bbox.y0 + h_height
        elif va == 'upper':
            y1_new = bbox.y1 
            y0_new = bbox.y1 - h_height
    elif loc == 'right':
        x0_new = bbox.x1 + pad
        x1_new = x0_new + h_width
        if va == 'lower':
            y0_new = bbox.y0 
            y1_new = bbox.y0 + h_height
        elif va == 'upper':
            y1_new = bbox.y1 
            y0_new = bbox.y1 - h_height
    elif loc == 'bottom':
        if ha =='left':
            x0_new = bbox.x0
            x1_new = bbox.x0 + v_width
        elif ha == 'right':
            x1_new = bbox.x1
            x0_new = bbox.x1 - v_width
        y0_new = bbox.y0 - pad - v_height
        y1_new = y0_new + v_height
    elif loc == 'top':
        if ha =='left':
            x0_new = bbox.x0
            x1_new = bbox.x0 + v_width
        elif ha == 'right':
            x1_new = bbox.x1
            x0_new = bbox.x1 - v_width
        elif ha == 'center':
            x1_new = bbox.x1 - width*0.5 + v_width*0.5 
            x0_new = bbox.x1 - width*0.5 - v_width*0.5 
        y0_new = bbox.y1 + pad
        y1_new = y0_new + v_height
    elif loc == 'lowerleft':
        x0_new = bbox.x0 - h_pad - h_width
        x1_new = x0_new + h_width
        y0_new = bbox.y0 - height - v_pad
        y1_new = y0_new + v_height
    elif loc == 'upperleft':
        x0_new = bbox.x0 - h_pad - h_width
        x1_new = x0_new + h_width
        y0_new = bbox.y1 + v_pad
        y1_new = y0_new + v_height
    elif loc == 'lowerright':
        x0_new = bbox.x1 + h_pad
        x1_new = x0_new + h_width
        y0_new = bbox.y0 - height - v_pad
        y1_new = y0_new + v_height
    elif loc == 'upperright':
        x0_new = bbox.x1 + h_pad
        x1_new = x0_new + h_width
        y0_new = bbox.y1 + v_pad
        y1_new = y0_new + v_height

    # Create new ax
    fig = axes[0].get_figure()
    bbox_new = mtransforms.Bbox.from_extents(x0_new, y0_new, x1_new, y1_new)
    ax_new = fig.add_axes(bbox_new)
    return ax_new

# %%
# plot_basic
import cartopy.crs as ccrs
import cartopy
import matplotlib.colors as mcolors

proj = ccrs.PlateCarree()

def plot_basic(var=None, loc_vmin=None, loc_vmax=None, loc_cbar_title='', ax=None, \
               loc_cmap_name='bwr', loc_cmap_lut=None, var_subset=None, lat_subset=None, lon_subset=None, \
                loc_s=2, loc_alpha=0.8, loc_linewidth=0.0, loc_edgecolor='k', loc_marker='o', \
                loc_cbar_on='horizontal', loc_norm_on=False, loc_vcenter=None, \
                powerlimits=False, loc_cbar_format=None, loc_title=None, loc_cbar_title_loc=None):
    
    if ax is None:
        # fig, ax = plt.subplots(1, 1, figsize=(4, 2), subplot_kw={'projection':ccrs.Robinson(central_longitude=0)}, dpi=300) # or 150
        fig, ax = plt.subplots(1, 1, figsize=(4, 2), subplot_kw={'projection':proj}, dpi=600)

    ax.set_extent([-180, 180, -90, 90])
    ax.set_title(loc_title, fontsize=7)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.,
        color='#bcbcbc',    # color
        linestyle='--',     # line stype
        x_inline = False,
        y_inline = False,
        xlocs = np.arange(-120, 180, 60),  # longitude line position
        ylocs = np.arange(-60, 90, 30),    # latitude line position
        # rotate_labels = False,           # rotate labels or not
        alpha = 0.3,                      # opacity of lines
        zorder=0,
    )
    gl.top_labels = False 
    gl.right_labels = False 
    ax.set_xticks(np.arange(-120, 180, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-60, 90, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_tick_params(width=0.7, length=3)
    ax.yaxis.set_tick_params(width=0.7, length=3)
    ax.tick_params(labelbottom=False, labelleft=False)

    
    ax.add_feature(cartopy.feature.LAND.with_scale('50m'), facecolor='#bcbcbc', edgecolor='none')

    loc_cmap = matplotlib.cm.get_cmap(loc_cmap_name, loc_cmap_lut) 

    if loc_norm_on and (loc_vcenter is not None) and (loc_vmin is not None) and (loc_vmax is not None):
        norm = mcolors.TwoSlopeNorm(vmin=loc_vmin, vcenter=loc_vcenter, vmax=loc_vmax)
        if var is not None:
            im1 = ax.scatter(
                x=lons,
                y=lats,
                c=var,
                cmap=loc_cmap,
                s=loc_s,
                alpha=loc_alpha,
                linewidth=loc_linewidth,
                marker=loc_marker,
                # linewidth=[0.02 if i<0.05 else np.nan for i in p_lswt_summ_yearmean],
                edgecolors=loc_edgecolor,
                norm=norm,
                # transform=ccrs.PlateCarree()
                )
        elif (var_subset is not None) and (lat_subset is not None) and (lon_subset is not None):
            im1 = ax.scatter(
                x=lon_subset,
                y=lat_subset,
                c=var_subset,
                cmap=loc_cmap,
                s=loc_s,
                alpha=loc_alpha,
                marker=loc_marker,
                linewidth=loc_linewidth,
                norm=norm,
                # linewidth=[0.02 if i<0.05 else np.nan for i in p_lswt_summ_yearmean],
                edgecolors=loc_edgecolor)
        else:
            print('no valid coordinates or data found')
    elif loc_norm_on == False:
        if var is not None:
            im1 = ax.scatter(
                x=lons,
                y=lats,
                c=var,
                cmap=loc_cmap,
                vmin=loc_vmin,
                vmax=loc_vmax,
                s=loc_s,
                alpha=loc_alpha,
                linewidth=loc_linewidth,
                marker=loc_marker,
                # linewidth=[0.02 if i<0.05 else np.nan for i in p_lswt_summ_yearmean],
                edgecolors=loc_edgecolor,
                # transform=ccrs.PlateCarree()
                )
        elif (var_subset is not None) and (lat_subset is not None) and (lon_subset is not None):
            im1 = ax.scatter(
                x=lon_subset,
                y=lat_subset,
                c=var_subset,
                cmap=loc_cmap,
                vmin=loc_vmin,
                vmax=loc_vmax,
                s=loc_s,
                alpha=loc_alpha,
                marker=loc_marker,
                linewidth=loc_linewidth,
                # linewidth=[0.02 if i<0.05 else np.nan for i in p_lswt_summ_yearmean],
                edgecolors=loc_edgecolor)  

    # if powerlimits:
    #     cbformat = matplotlib.ticker.ScalarFormatter()
    #     cbformat.set_powerlimits((-4,12))
    #     cbformat.set_useMathText(True)
        
    if loc_cbar_on == 'vertical':
        cbax = add_equal_axes(ax, 'right', 0.03, h_width=0.015)
        if powerlimits:
            if loc_cbar_title_loc == 'top':
                cb = plt.colorbar(im1, cax=cbax, ax=ax, fraction=0.019, format=loc_cbar_format)
                cb.ax.set_title(loc_cbar_title, fontsize=7)
            else:
                cb = plt.colorbar(im1, cax=cbax, ax=ax, fraction=0.019, label=loc_cbar_title, format=loc_cbar_format)
            cb.formatter.set_powerlimits((-2,12))
            cb.formatter.set_useMathText(True)
            # 这会让科学计数法和loc_cbar_format失效。
            if loc_norm_on:
                cb.ax.set_yscale('linear')
        else:
            if loc_cbar_title_loc == 'top':
                cb = plt.colorbar(im1, cax=cbax, ax=ax, fraction=0.019, format=loc_cbar_format)
                cb.ax.set_title(loc_cbar_title, fontsize=7)
            else:
                cb = plt.colorbar(im1, cax=cbax, ax=ax, fraction=0.019, label=loc_cbar_title, format=loc_cbar_format)
            if loc_norm_on:
                cb.ax.set_yscale('linear')
        return cbax
    elif loc_cbar_on == 'horizontal':
        # cbax = add_equal_axes(ax, 'bottom', 0.15, v_height=0.04)
        # cbax = add_equal_axes(ax, 'bottom', 0.03, v_height=0.012)
        cbax = add_equal_axes(ax, 'bottom', 0.05, v_height=0.02)
        cb = plt.colorbar(im1, cax=cbax, ax=ax, fraction=0.019, label=loc_cbar_title, orientation='horizontal', format=loc_cbar_format)
        cb.ax.set_xscale('linear')
        return cbax
    elif loc_cbar_on == False:
        return im1

# plot_basic(fluxes_slope['sh'], loc_vcenter=0, loc_cbar_on='vertical', loc_cbar_title='test', loc_cbar_title_loc='top')
# plot_basic(slope_lswt_ctl_cov, loc_cbar_on='vertical', powerlimits=True, loc_vmin=-0.0001, loc_vmax=0.0001)
# plot_basic(var_subset=contribution_summ[filter1], lat_subset=lats[filter1], lon_subset=lons[filter1], loc_marker='o', loc_cmap='coolwarm', loc_vmin=-100, loc_vmax=100)

# %%
# plot_scatter
from scipy.stats import gaussian_kde

def myNormalize(x):
    return (x-min(x))/(max(x)-min(x))

def plot_scatter(loc_ax, obs, sim, default=None, loc_title=None, loc_unit='\u00B0C', \
    loc_ylabel='Observation', loc_xlabel='CSFLake', \
    loc_label=None, loc_min=0, loc_max=35, loc_xy_spacing=10, loc_s=20, loc_cbar_format=None, \
        loc_cbar_on=None, loc_cbar_norm=True, loc_vmin=None, loc_vmax=None):
    
    if loc_ax is None:
        fig, loc_ax = plt.subplots(figsize=(3, 2.5), dpi=300)

    mae = np.nanmean(np.abs(sim - obs))
    # rmse = np.sqrt(np.nanmean((sim - obs) ** 2))
    cc = np.corrcoef(obs, sim)[0,1]
    # r2 = r2_score(obs, sim)
    # print(cc, rmse)
    xy = np.vstack([obs,sim])
    z = gaussian_kde(xy)(xy)
    if loc_cbar_norm:
        z = myNormalize(z)
    
    im = loc_ax.scatter(sim, obs, c=z, cmap='Spectral_r', alpha=0.7, vmin=loc_vmin, vmax=loc_vmax, s=loc_s, linewidths=0)
    if default:
        im2 = loc_ax.scatter(default, obs, edgecolors='k', linewidths=0.5, c='white')
    # loc_ax.set_xticks(np.arange(loc_min,loc_max+loc_xy_spacing,loc_xy_spacing))
    # loc_ax.set_yticks(np.arange(loc_min,loc_max+loc_xy_spacing,loc_xy_spacing))
    loc_ax.set_xlim([loc_min, loc_max])
    loc_ax.set_ylim([loc_min, loc_max])
    loc_ax.set_ylabel(loc_ylabel+' ({})'.format(loc_unit))
    loc_ax.set_xlabel(loc_xlabel+' ({})'.format(loc_unit))

    if loc_cbar_on == 'inside':
        cbar=plt.colorbar(
            im,
            shrink=1,
            orientation='vertical',
            extend='neither',
            ax=loc_ax,
            format=loc_cbar_format,
            pad=0.03, fraction=0.05, # change fraction with figure size
            )
        cbar.ax.set_title('Density')
    elif loc_cbar_on == True:
        pos_ax = loc_ax.get_position()
        cbaxes = loc_ax.get_figure().add_axes([(pos_ax.x0+pos_ax.width)*1.02, pos_ax.y0, 0.06*pos_ax.width, pos_ax.height]) #Add position (left, bottom, width, height)
        cbar = loc_ax.get_figure().colorbar(
            im, 
            ax=loc_ax,
            cax=cbaxes,
            orientation='vertical', 
            extend='neither',
            pad=0.04, 
            fraction=0.037, 
            label='Density'
            )     # rmse
    # for normal scatter plot
    # pad=0.015,
    # aspect=30,
    loc_ax.plot((0, 1), (0, 1), transform=loc_ax.transAxes, ls='--', label="1:1 line", c='silver', zorder=0)
    loc_ax.text(0.05, 0.9, '$n\ =\ {}$'.format(len(obs)), transform=loc_ax.transAxes)
    # loc_ax.text(0.05, 0.8, '$r\ =\ {:.2f}$'.format(cc), transform=loc_ax.transAxes)
    loc_ax.text(0.05, 0.8, 'MAE = {:.2f} {}'.format(mae, loc_unit), transform=loc_ax.transAxes)
    # loc_ax.text(0.05, 0.7, 'RMSE = {:.2f} {}'.format(rmse, loc_unit), transform=loc_ax.transAxes)
    # loc_ax.text(0.05, 0.6, '$R^2\ =\ {:.2f}$'.format(r2), transform=loc_ax.transAxes)
    loc_ax.set_title(loc_title)

    return im 

# %%
def auto_label(*args, x_offset=0, y_offset=0):
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    for i, ax in enumerate(np.array(args).flatten()):
        ax.annotate('({})'.format(labels[i]), xy=(0+x_offset, 1.03+y_offset), xycoords='axes fraction', weight="bold")  

# %%
# def plot_scenario
ssp_colors = [None]*4   # 'SSP1-2.6','SSP3-7.0','SSP5-8.5', historical
ssp_colors[0] = plt.cm.get_cmap('tab20c').colors[0]
# ssp_colors[1] = plt.cm.get_cmap('tab20c').colors[8]
ssp_colors[1] = plt.cm.get_cmap('tab20c').colors[4]
ssp_colors[2] = plt.cm.get_cmap('tab20c').colors[12]
ssp_colors[3] = plt.cm.get_cmap('tab20c').colors[16]
# ivar = 'total_days'
def plot_scenario(ax, ivar, lw_ratio=1, legend_loc=None, legend_on=True):
    # fig, ax = plt.subplots()
    # hist and obs
    x_hist = np.arange(1979, 2023)
    # hist
    mme = hw_future_mme_global.sel(var=ivar, time=slice('1979', '2022'), scenario='ssp126')
    ax.plot(x_hist, mme, c=ssp_colors[-1], label='Historical (ISIMIP3b)', lw=1.5*lw_ratio)
    std = hw_future_std_global.sel(var=ivar, time=slice('1979', '2022'), scenario='ssp126')
    low_bound = mme - std
    upp_bound = mme + std 
    ax.fill_between(x_hist, low_bound, upp_bound, color=ssp_colors[-1], alpha=0.2, lw=0)
    # obs
    # note that df_xxx should also exclude these lakes, which will be done in the function plot_scenarios1
    ax.plot(x_hist, globals()['df_'+ivar+'_mean_future'], c='k', ls='--', lw=1.8*lw_ratio, label='Historical (ERA5-Land)')
    # future
    x_future = np.arange(2022, 2101)
    legends = ['SSP1-2.6', 'SSP3-7.0', 'SSP5-8.5']
    for i, ssp in enumerate(['ssp126', 'ssp370', 'ssp585']):
        mme = hw_future_mme_global.sel(var=ivar, time=slice('2022', '2100'), scenario=ssp)
        ax.plot(x_future, mme, c=ssp_colors[i], label=legends[i], lw=1.5*lw_ratio)
        std = hw_future_std_global.sel(var=ivar, time=slice('2022', '2100'), scenario=ssp)
        low_bound = mme - std
        upp_bound = mme + std 
        ax.fill_between(x_future, low_bound, upp_bound, color=ssp_colors[i], alpha=0.2, lw=0)
    if legend_on:
        ax.legend(frameon=False, bbox_to_anchor=legend_loc)
    return ax

# %% [markdown]
# # Load data 

# %%
# Note: set as the location of PaperData
datadir = './PaperData/'

# %%
cali_lake = pd.read_csv(datadir + 'LakeInformation.csv', index_col=0)
cali_lakeids = cali_lake.index.to_numpy()    # Hylak_id 
lats = cali_lake['centroid_y'].to_numpy()    
lons = cali_lake['centroid_x'].to_numpy()
elevation = cali_lake['Elevation'].to_numpy()
depth = cali_lake['Depth_avg'].to_numpy()
area = cali_lake['Lake_area'].to_numpy()
# mask of northern/southern hemisphere
cali_mask_north = np.where(lats>0)[0]
cali_mask_south = np.where(lats<0)[0]

# %%
with open(datadir + 'study_sites.pkl', 'rb') as f:
    tmpdf_ef1 = pickle.load(f)

# %%
with open(datadir + 'lswt_validation.pkl', 'rb') as f:
    deflswt_metrics_c, deflswt_metrics_v, lswt_metrics_c, lswt_metrics_v, globolakes_metric, total_insitu_sim = pickle.load(f)

# %%
with open(datadir + 'lhw_validation.pkl', 'rb') as f:
      corr_90thresh, corr_seasmean, mae_90thresh, mae_seasmean, mae_hw_clim, tested_lakes, stack_satt, stack_sim, hw_satt, hw_sim = pickle.load(f)

# %%
with open(datadir + 'trend.pkl', 'rb') as f:
    df_slope, df_p = pickle.load(f)

# %%
with open(datadir + 'annual_statistics.pkl', 'rb') as f:
    df_rate_onset_relThresh_mean, df_rate_decline_relThresh_mean, \
    df_reaction_window_relThresh_duration_mean, df_coping_window_relThresh_duration_mean, \
    df_recovery_window_duration_mean, df_recovery_window_between_years, \
    df_count_mean, df_intensity_mean_mean, df_duration_mean, \
    number_of_lakes_with_lhw = pickle.load(f)

# %%
with open(datadir + 'climatology_mean.pkl', 'rb') as f:
    df_rate_onset_relThresh_clim, df_rate_decline_relThresh_clim, \
    df_reaction_window_relThresh_duration_clim, df_coping_window_relThresh_duration_clim, \
    df_recovery_window_duration_clim = pickle.load(f)

# %%
with open(datadir + 'local_drivers.pkl', 'rb') as f:
    change_ratio_onset_stats, change_ratio_decline_stats, min_onset_values, min_decline_values, \
    min_onset_string, min_decline_string, bx_sh, bx_lh, bx_netlw, bx_mlsw = pickle.load(f)

# %%
with gzip.GzipFile(datadir + 'future.pkl.gzip', 'rb') as f:
    hw_future_clim_diff, hw_future_mme_global, hw_future_std_global, include, exclude, \
    stack_ctl, stack_gfdl, stack_ipsl, stack_mri, stack_ukesm1, \
    df_rate_onset_relThresh_mean_future, df_rate_decline_relThresh_mean_future, \
    df_reaction_window_relThresh_duration_mean_future, df_coping_window_relThresh_duration_mean_future, \
    df_recovery_window_duration_mean_future, df_recovery_window_between_years_mean_future, \
    df_intensity_max_relThresh_mean_future = pickle.load(f)

# %%
with open(datadir + 'future_detrended.pkl', 'rb') as f:
    hw_future_detrended_mme_global, hw_future_detrended_std_global = pickle.load(f)

# %% [markdown]
# # Figures

# %% [markdown]
# ### F1
# Rate of onset/decline trend
# Reaction/coping/recovery window trend

# %%
fig = plt.figure(figsize=(18/2.54, 25/2.54), dpi=500)
plt.subplots_adjust(wspace=0.2, hspace=0.25)
ax2 = plt.subplot(522, projection=proj)
ax4 = plt.subplot(524, projection=proj)
ax6 = plt.subplot(526, projection=proj)
ax8 = plt.subplot(528, projection=proj)
ax10 = plt.subplot(5,2,10, projection=proj)

lw = 0.0 
plot_basic(ax=ax2, var=df_slope['rate_onset_relThresh'], loc_linewidth=lw, loc_vmin=-0.15, loc_vmax=0.15, loc_s=2, \
           loc_cbar_on='vertical', loc_cbar_title='Rate of onset trend\n(\u00B0C d$^{-1}$ 10a$^{-1}$)')

plot_basic(ax=ax4, var=df_slope['rate_decline_relThresh'], loc_linewidth=lw, loc_vmin=-0.25, loc_vmax=0.25, loc_s=2, \
           loc_cbar_on='vertical', loc_cbar_title='Rate of decline trend\n(\u00B0C d$^{-1}$ 10a$^{-1}$)')

ax1 = add_equal_axes(ax2, 'left', 0.05, h_width=None)
tmpdf = df_rate_onset_relThresh_mean
tmpmk = mk.original_test(tmpdf)
ax1.plot(np.arange(1979,2023), tmpdf, linewidth=1, c='#5770db')
ax1.plot(np.arange(1979,2023), tmpmk[7]*np.arange(0,2023-1979)+tmpmk[8], alpha=0.45, linewidth=2.2, c='#5770db')
ax1.set_ylabel('Rate of onset (\u00B0C d$^{-1}$)',)
ax1.annotate(
    '{:+.2f} '.format(tmpmk[7]*10) + '\u00B0C d$^{-1}$ 10a$^{-1},\ p$ < 0.005',
    xy=(0.36, 0.05), xycoords='axes fraction',
    color='k'
    )
ax3 = add_equal_axes(ax4, 'left', 0.05, h_width=None)
tmpdf = df_rate_decline_relThresh_mean
tmpmk = mk.original_test(tmpdf)
ax3.plot(np.arange(1979,2023), tmpdf, linewidth=1, c='#5770db')
ax3.plot(np.arange(1979,2023), tmpmk[7]*np.arange(0,2023-1979)+tmpmk[8], alpha=0.45, linewidth=2.2, c='#5770db')
ax3.set_ylabel('Rate of decline (\u00B0C d$^{-1}$)',)
ax3.annotate(
    '{:+.2f} '.format(tmpmk[7]*10) + '\u00B0C d$^{-1}$ 10a$^{-1},\ p$ < 0.005',
    xy=(0.36, 0.05), xycoords='axes fraction',
    color='k'
    )

ax1.set_ylim([0.15, 0.45])
ax3.set_ylim([0.22, 0.58])

plot_basic(ax=ax6, var=df_slope['reaction_window_relThresh_duration'], loc_linewidth=lw, loc_vmin=-2, loc_vmax=2, \
           loc_s=2, loc_cbar_on='vertical', loc_cbar_title='Reaction window duration trend\n(d$^{-1}$ 10a$^{-1}$)')

plot_basic(ax=ax8, var=df_slope['coping_window_relThresh_duration'], loc_linewidth=lw, loc_vmin=-2, loc_vmax=2, \
           loc_s=2, loc_cbar_on='vertical', loc_cbar_title='Coping window duration trend\n(d$^{-1}$ 10a$^{-1}$)')

plot_basic(df_slope['recovery_window_duration'], loc_linewidth=lw, loc_vmin=-20, loc_vmax=20, ax=ax10, \
           loc_cbar_on='vertical', loc_cbar_title='Recovery window duration trend\n(d 10a$^{-1}$)')

ax5 = add_equal_axes(ax6, 'left', 0.05, h_width=None)
tmpdf = df_reaction_window_relThresh_duration_mean
tmpmk = mk.original_test(tmpdf)
ax5.plot(np.arange(1979,2023), tmpdf, linewidth=1, c='#5770db')
ax5.plot(np.arange(1979,2023), tmpmk[7]*np.arange(0,2023-1979)+tmpmk[8], alpha=0.45, linewidth=2.2, c='#5770db')
ax5.set_ylabel('Reaction window duration (d)',)
ax5.annotate(
    '{:+.2f} '.format(tmpmk[7]*10) + 'd 10d$^{-1},\ p$ < 0.005',
    xy=(0.45, 0.05), xycoords='axes fraction',
    color='k'
    )

ax7 = add_equal_axes(ax8, 'left', 0.05, h_width=None)
tmpdf = df_coping_window_relThresh_duration_mean
tmpmk = mk.original_test(tmpdf)
ax7.plot(np.arange(1979,2023), tmpdf, linewidth=1, c='#5770db')
ax7.plot(np.arange(1979,2023), tmpmk[7]*np.arange(0,2023-1979)+tmpmk[8], alpha=0.45, linewidth=2.2, c='#5770db')
ax7.set_ylabel('Coping window duration (d)',)
ax7.annotate(
    '{:+.2f} '.format(tmpmk[7]*10) + 'd 10a$^{-1},\ p$ < 0.005',
    xy=(0.45, 0.05), xycoords='axes fraction',
    color='k'
    )

ax9 = add_equal_axes(ax10, 'left', 0.05, h_width=None)
tmpdf = df_recovery_window_duration_mean
tmpmk = mk.original_test(tmpdf)
print(tmpmk[2], tmpmk[7])
ax9.plot(np.arange(1979,2023), tmpdf, linewidth=1, c='#5770db')
ax9.plot(np.arange(1979,2023), tmpmk[7]*np.arange(0,2023-1979)+tmpmk[8], alpha=0.45, linewidth=2.2, c='#5770db')
ax9.set_ylabel('Recovery window duration (d)',)
ax9.annotate(
    '${:+.2f}$ '.format(tmpmk[7]*10) + 'd 10a$^{-1},\ p$ < 0.01',
    xy=(0.45, 0.85), xycoords='axes fraction',
    color='k'
    )

auto_label(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10)

# %% [markdown]
# ### F2
# air-lake heat fluxes contribution

# %%
def plot_fig3bc(var=None, loc_lats=lats, loc_lons=lons, ax=None, loc_s=3, loc_c='k',\
    loc_alpha=0.8, loc_linewidth=0.0, loc_marker='o', loc_title=None):
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12/2.54, 6/2.54), subplot_kw={'projection':proj}, dpi=600)

    ax.set_extent([-180, 180, -90, 90])
    ax.add_feature(cartopy.feature.LAND.with_scale('50m'), facecolor='#bcbcbc', edgecolor='none')
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.,
        color='#bcbcbc',    # color
        linestyle='--',     # line stype
        x_inline = False,
        y_inline = False,
        xlocs = np.arange(-180, 180, 60),  # longitude line position
        ylocs = np.arange(-90, 90, 30),    # latitude line position
        # rotate_labels = False,           # rotate labels or not
        alpha = 0.3,                      # opacity of lines
        zorder=0,
    )
    gl.top_labels = False 
    gl.right_labels = False 
    ax.set_xticks(np.arange(-120, 180, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-60, 90, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_tick_params(width=0.7, length=3)
    ax.yaxis.set_tick_params(width=0.7, length=3)
    ax.tick_params(labelbottom=False, labelleft=False)
       
    im1 = ax.scatter(
        x=loc_lons,
        y=loc_lats,
        c=loc_c,
        s=loc_s,
        alpha=loc_alpha,
        linewidth=loc_linewidth,
        marker=loc_marker,
        )
    
    # Use color to distingush onset/decline
    # lh #E6639B, sh #9D47B6, netlw #0FBDE9, mlsw #2F3BAD 

    legend_lake_info = [
        Line2D([0], [0], marker='o', color='none', 
            markerfacecolor='#E6639B', 
            markeredgewidth=0., markersize=3,
            label='Sensible heat flux'),
        Line2D([0], [0], marker='o', color='none', 
            markerfacecolor='#9D47B6', 
            markeredgewidth=0., markersize=3,
            label='Latent heat flux'),
        Line2D([0], [0], marker='o', color='none', 
            markerfacecolor='#0FBDE9', 
            markeredgewidth=0., markersize=3,
            label='Longwave radiation'),
        Line2D([0], [0], marker='o', color='none', 
            markerfacecolor='#2F3BAD', 
            markeredgewidth=0., markersize=3,
            label='Shortwave radiation'),
        ]
    ax.legend(handles=legend_lake_info, loc='center', bbox_to_anchor=(0.5,-0.26), frameon=False, ncol=2)
    ax.set_title(loc_title)

c3 = np.array(['#E6639B'] * len(min_onset_string))
c3[min_onset_string=='rmse_lh'] = '#9D47B6'
c3[min_onset_string=='rmse_netlw'] = '#0FBDE9'
c3[min_onset_string=='rmse_mlsw'] = '#2F3BAD'

c4 = np.array(['#E6639B'] * len(min_decline_string))
c4[min_decline_string=='rmse_lh'] = '#9D47B6'
c4[min_decline_string=='rmse_netlw'] = '#0FBDE9'
c4[min_decline_string=='rmse_mlsw'] = '#2F3BAD'

fig = plt.figure(figsize=(18/2.54, 12/2.54), dpi=500)
plt.subplots_adjust(wspace=0.2, hspace=0.25)
ax1 = plt.subplot(221, projection=proj)
ax2 = plt.subplot(222, projection=proj)
ax3 = plt.subplot(223, projection=proj)
ax4 = plt.subplot(224, projection=proj)

plot_basic(change_ratio_onset_stats['change_ratio'], ax=ax1, loc_s=2, loc_vmin=0.5, loc_vmax=1.5, loc_cmap_name='plasma_r', loc_cbar_on='horizontal', loc_cbar_title='Change ratio', loc_title='Onset')
plot_basic(change_ratio_decline_stats['change_ratio'], ax=ax2, loc_s=2, loc_vmin=0.5, loc_vmax=1.5, loc_cmap_name='plasma_r', loc_cbar_on='horizontal', loc_cbar_title='Change ratio', loc_title='Decline')

plot_fig3bc(ax=ax3, var=min_onset_values, loc_c=c3, loc_s=2,)
plot_fig3bc(ax=ax4, var=min_decline_values, loc_c=c4, loc_s=2)

auto_label(ax1, ax2, ax3, ax4)

# %% [markdown]
# # Supplementary Figures 

# %% [markdown]
# ### F3
# Future rate of onset/decline global temporal changes + differences of time slice

# %%
fig = plt.figure(figsize=(16/2.54, 25/2.54), dpi=500)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
ax2 = plt.subplot(511, projection=proj)
ax4 = plt.subplot(512, projection=proj)
ax6 = plt.subplot(513, projection=proj)
ax8 = plt.subplot(514, projection=proj)
ax10 = plt.subplot(515, projection=proj)

im2 = plot_basic(var_subset=hw_future_clim_diff.sel(var='rate_onset_relThresh', scenario='ssp585'), \
           lat_subset=lats[include], lon_subset=lons[include], ax=ax2, loc_vmin=-0.6, loc_vmax=0.6, loc_cbar_on='vertical')

plot_basic(var_subset=hw_future_clim_diff.sel(var='rate_decline_relThresh', scenario='ssp585'), \
           lat_subset=lats[include], lon_subset=lons[include], ax=ax4, loc_vmin=-0.6, loc_vmax=0.6, loc_cbar_on='vertical')

im6 = plot_basic(var_subset=hw_future_clim_diff.sel(var='reaction_window_relThresh_duration', scenario='ssp585'), \
           lat_subset=lats[include], lon_subset=lons[include], ax=ax6, loc_vmin=-20, loc_vmax=80, loc_vcenter=0, loc_norm_on=True, \
            loc_cbar_on='vertical')

plot_basic(var_subset=hw_future_clim_diff.sel(var='coping_window_relThresh_duration', scenario='ssp585'), \
           lat_subset=lats[include], lon_subset=lons[include], ax=ax8, loc_vmin=-50, loc_vmax=150, loc_vcenter=0, loc_norm_on=True, \
            loc_cbar_on='vertical')

plot_basic(var_subset=hw_future_clim_diff.sel(var='recovery_window_duration', scenario='ssp585'), \
           lat_subset=lats[include], lon_subset=lons[include], ax=ax10, loc_vmin=-80, loc_vmax=20, loc_vcenter=0, loc_norm_on=True, \
            loc_cbar_on='vertical')

ax1 = add_equal_axes(ax2, 'left', pad=0.08, h_width=0.3)
plot_scenario(ax1, 'rate_onset_relThresh', lw_ratio=0.6, legend_on=True)

ax3 = add_equal_axes(ax4, 'left', pad=0.08, h_width=0.3)
plot_scenario(ax3, 'rate_decline_relThresh', lw_ratio=0.6, legend_on=False)

ax5 = add_equal_axes(ax6, 'left', pad=0.08, h_width=0.3)
plot_scenario(ax5, 'reaction_window_relThresh_duration', lw_ratio=0.6, legend_on=False)

ax7 = add_equal_axes(ax8, 'left', pad=0.08, h_width=0.3)
plot_scenario(ax7, 'coping_window_relThresh_duration', lw_ratio=0.6, legend_on=False)

ax9 = add_equal_axes(ax10, 'left', pad=0.08, h_width=0.3)
plot_scenario(ax9, 'recovery_window_duration', lw_ratio=0.6, legend_on=False)

ax1.set_ylabel('Rate of onset (\u00B0C d$^{-1}$)')
ax3.set_ylabel('Rate of decline (\u00B0C d$^{-1}$)')
ax5.set_ylabel('Reaction window duration (d)')
ax7.set_ylabel('Coping window duration (d)')
ax9.set_ylabel('Recovery window duration (d)')

auto_label(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10)

# %% [markdown]
# ### SF2
# Study sites

# %%
# plot_extFig1
def plot_extFig1(var=None, loc_lats=lats, loc_lons=lons, ax=None, loc_s=3, loc_c='k', loc_cmap_name='viridis', \
    loc_alpha=0.8, loc_linewidth=0.0, loc_marker='o', loc_vmin=None, loc_vmax=None, loc_norm=None):
    ax.set_extent([-180, 180, -90, 90])
    ax.add_feature(cartopy.feature.LAND.with_scale('50m'), facecolor='#bcbcbc', edgecolor='none')
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.,
        color='#bcbcbc',    # color
        linestyle='--',     # line stype
        x_inline = False,
        y_inline = False,
        xlocs = np.arange(-180, 180, 60),  # longitude line position
        ylocs = np.arange(-90, 90, 30),    # latitude line position
        # rotate_labels = False,           # rotate labels or not
        alpha = 0.3,                      # opacity of lines
        zorder=0,
    )
    gl.top_labels = False 
    gl.right_labels = False 
    ax.set_xticks(np.arange(-120, 180, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-60, 90, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_tick_params(width=0.7, length=3)
    ax.yaxis.set_tick_params(width=0.7, length=3)
    ax.tick_params(labelbottom=False, labelleft=False)
    
    im1 = ax.scatter(
        x=loc_lons,
        y=loc_lats,
        c=loc_c,
        s=loc_s,
        alpha=loc_alpha,
        linewidth=loc_linewidth,
        marker=loc_marker,
        cmap=loc_cmap_name,
        vmin=loc_vmin,
        vmax=loc_vmax,
        norm=loc_norm,
        )
    return im1 

# %%
fig = plt.figure(figsize=(14/2.54, 10/2.54), dpi=500)
ax = plt.subplot(111, projection=proj)
im1 = plot_extFig1(cali_lakeids, ax=ax, loc_c=depth, loc_norm=matplotlib.colors.LogNorm(), loc_cmap_name='cmr.cosmic_r')
cbax = add_equal_axes(ax, 'bottom', 0.1, v_height=0.025)
cb = plt.colorbar(im1, cax=cbax, ax=ax, fraction=0.019, orientation='horizontal')
cb.ax.set_title('Depth (m)', fontsize=7)

ax2 = add_equal_axes(ax, 'top', 0.06, v_height=0.2)
ax2.spines[['top', 'right']].set_visible(False)
ax2.bar(tmpdf_ef1.sort_values('lonBin')['lonBin'], tmpdf_ef1.sort_values('lonBin')['area'], linewidth=0, color='gray', width=1)
ax2.set_xlim([-180, 180])
ax2.tick_params(bottom=False, labelbottom=False)
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Surface area (km$^2$)')

ax3 = add_equal_axes(ax, 'right', 0.06, h_width=0.16)
ax3.spines[['top', 'right']].set_visible(False)
ax3.barh(tmpdf_ef1.sort_values('latBin')['latBin'], tmpdf_ef1.sort_values('latBin')['area'], linewidth=0, color='gray', height=1)
ax3.set_ylim([-90, 90])
ax3.tick_params(left=False, labelleft=False)
ax3.set_ylabel('Latitude')
ax3.set_xlabel('Surface area (km$^2$)')

auto_label(ax2, ax, ax3)

# %% [markdown]
# ### SF3
# LSWT validation 

# %%
# start plotting
fig = plt.figure(figsize=(16/2.54, 15/2.54))
plt.subplots_adjust(hspace=0.2)
ax1 = plt.subplot(321, projection=proj)
ax2 = plt.subplot(322, projection=proj)
ax3 = plt.subplot(323, projection=proj)
ax4 = plt.subplot(324, projection=proj)
ax5 = plt.subplot(325, projection=proj)

im1 = plot_basic(deflswt_metrics_c['mae'], ax=ax1, loc_cmap_name='viridis_r', loc_vmin=0, loc_vmax=5, loc_cmap_lut=None, loc_title='FLake vs. Lakes_cci (2001\u20132010)', loc_cbar_on=False)
plot_basic(deflswt_metrics_v['mae'], ax=ax2, loc_cmap_name='viridis_r', loc_vmin=0, loc_vmax=5, loc_cmap_lut=None, loc_title='FLake vs. Lakes_cci (2011\u20132020)', loc_cbar_on=False)
im3 = plot_basic(lswt_metrics_c['mae'], ax=ax3, loc_cmap_name='viridis_r', loc_vmin=0, loc_vmax=2.5, loc_cmap_lut=None, loc_title='CSFLake vs. Lakes_cci (2001\u20132010)', loc_cbar_on=False)
plot_basic(lswt_metrics_v['mae'], ax=ax4, loc_cmap_name='viridis_r', loc_vmin=0, loc_vmax=2.5, loc_cmap_lut=None, loc_title='CSFLake vs. Lakes_cci (2011\u20132020)', loc_cbar_on=False)

cbax = add_equal_axes([ax1, ax2], 'right', 0.03, h_width=0.015)
cb = plt.colorbar(im1, cax=cbax, ax=ax1, fraction=0.019,)
cb.ax.set_title('MAE (\u00B0C)', fontsize=7)

plot_basic(globolakes_metric['mae'], ax=ax5, loc_cmap_name='viridis_r', loc_vmin=0, \
           loc_vmax=2.5, loc_cbar_on='vertical', loc_cbar_title='MAE (\u00B0C)', loc_cbar_title_loc='top',\
              loc_title='CSFLake vs. GloboLakes')

cbax = add_equal_axes([ax3, ax4], 'right', 0.03, h_width=0.015)
cb = plt.colorbar(im3, cax=cbax, ax=ax3, fraction=0.019,)
cb.ax.set_title('MAE (\u00B0C)', fontsize=7)

ax6 = add_equal_axes(ax5, 'right', 0.16, h_width=0.262)
im = plot_scatter(obs=total_insitu_sim['obsTemp'], sim=total_insitu_sim['simTemp'], loc_ax=ax6, loc_s=5, loc_cbar_on=False)
ax6.set_title('CSFLake vs. $in\ situ$', fontsize=7)

cbax = add_equal_axes(ax6, 'right', 0.03, h_width=0.015)
cb = plt.colorbar(im, cax=cbax, ax=ax3, fraction=0.019,)
cb.ax.set_title('Density', fontsize=7)

auto_label(ax1, ax2, ax3, ax4, ax5, ax6)

# %% [markdown]
# ### SF4
# Threshold and climatology validation

# %%
fig = plt.figure(figsize=(18/2.54, 10/2.54), dpi=500)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
ax1 = plt.subplot(221, projection=proj)
ax2 = plt.subplot(222, projection=proj)
ax3 = plt.subplot(223, projection=proj)
ax4 = plt.subplot(224, projection=proj)

im2 = plot_basic(corr_90thresh, ax=ax1, loc_vmin=0.9, loc_vmax=1, loc_linewidth=0.0, loc_cmap_name='coolwarm', loc_title='90th percentile', loc_cbar_on=False)
plot_basic(corr_seasmean, ax=ax2, loc_vmin=0.9, loc_vmax=1, loc_linewidth=0.0, loc_cmap_name='coolwarm', loc_title='Climatological mean', loc_cbar_on=False)
cbax = add_equal_axes([ax1, ax2], 'right', 0.03, h_width=0.015)
cb = plt.colorbar(im2, cax=cbax, ax=ax2, fraction=0.019,)
cb.ax.set_title('Correlation coefficient', fontsize=7)

im1 = plot_basic(mae_90thresh, ax=ax3, loc_vmin=0, loc_vmax=2, loc_linewidth=0.0, loc_cmap_name='viridis_r', loc_title='90th percentile (\u00B0C)', loc_cbar_on=False, loc_cbar_title='MAE (\u00B0C)')
plot_basic(mae_seasmean, ax=ax4, loc_vmin=0, loc_vmax=2, loc_linewidth=0.0, loc_cmap_name='viridis_r', loc_title='Climatological mean (\u00B0C)', loc_cbar_on=False, loc_cbar_title='MAE (\u00B0C)')
cbax = add_equal_axes([ax3, ax4], 'right', 0.03, h_width=0.015)
cb = plt.colorbar(im1, cax=cbax, ax=ax3, fraction=0.019,)
cb.ax.set_title('MAE', fontsize=7)

auto_label(ax1, ax2, ax3, ax4)

# %% [markdown]
# ### SF8 

# %%
fig = plt.figure(figsize=(18/2.54, 16/2.54), dpi=600)
plt.subplots_adjust(wspace=0.2, hspace=0.2)
ax1 = plt.subplot(321, projection=proj)
ax2 = plt.subplot(322, projection=proj)
ax3 = plt.subplot(323, projection=proj)
ax4 = plt.subplot(324, projection=proj)
ax5 = plt.subplot(325, projection=proj)

plot_basic(ax=ax1, var=df_rate_onset_relThresh_clim, loc_cmap_name='cmr.bubblegum_r', loc_alpha=0.9, loc_vmin=0, \
           loc_vmax=1.2, loc_cbar_on=False, loc_title='Mean rate of onset (\u00B0C d$^{-1}$)')
plot_basic(ax=ax2, var=df_rate_decline_relThresh_clim, loc_cmap_name='cmr.bubblegum_r', loc_alpha=0.9, loc_vmin=0, \
           loc_vmax=1.2, loc_cbar_on='vertical', loc_title='Mean rate of decline (\u00B0C d$^{-1}$)')
plot_basic(ax=ax3, var=df_reaction_window_relThresh_duration_clim, loc_cmap_name='cmr.bubblegum_r', loc_alpha=0.9, \
           loc_vmin=1, loc_vmax=7, loc_cbar_on=False, loc_title='Mean reaction window duration (d)')
plot_basic(ax=ax4, var=df_coping_window_relThresh_duration_clim, loc_cmap_name='cmr.bubblegum_r',loc_alpha=0.9, \
           loc_vmin=1, loc_vmax=7, loc_cbar_on='vertical', loc_title='Mean coping window duration (d)')
plot_basic(ax=ax5, var=df_recovery_window_duration_clim, loc_cmap_name='cmr.bubblegum_r', loc_alpha=0.9, \
           loc_vmin=0, loc_vmax=100, loc_cbar_on='vertical', loc_title='Mean recovery window duration (d)')

auto_label(ax1, ax2, ax3, ax4, ax5)

# %% [markdown]
# ### SF9

# %%
fig, axes = plt.subplots(2, 3, figsize=(18/2.54, 11/2.54), sharey=False, sharex=True)

plt.subplots_adjust(wspace=0.25, hspace=0.6)  # put this before create cbax!

axes.flatten()[-1].axis('off')
for ax in axes.reshape(-1):
    ax.spines[['right', 'top']].set_visible(False)
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.set_xlabel('Depth (log10; m)')

axes[0,0].scatter(y=df_rate_onset_relThresh_clim, x=np.log10(depth), s=5, alpha=0.6, lw=0, c='#4c72b0')
axes[0,1].scatter(y=df_rate_decline_relThresh_clim, x=np.log10(depth), s=5, alpha=0.6, lw=0, c='#4c72b0')
axes[0,2].scatter(y=df_reaction_window_relThresh_duration_clim, x=np.log10(depth), s=5, alpha=0.6, lw=0, c='#4c72b0')
axes[1,0].scatter(y=df_coping_window_relThresh_duration_clim, x=np.log10(depth), s=5, alpha=0.6, lw=0, c='#4c72b0')
axes[1,1].scatter(y=df_recovery_window_duration_clim, x=np.log10(depth), s=5, alpha=0.6, lw=0, c='#4c72b0')

axes[0,0].set_title('Mean rate of onset (\u00B0C d$^{-1}$)', fontsize=7)
axes[0,1].set_title('Mean rate of decline (\u00B0C d$^{-1}$)', fontsize=7)
axes[0,2].set_title('Mean reaction window duration (d)', fontsize=7)
axes[1,0].set_title('Mean coping window duration (d)', fontsize=7)
axes[1,1].set_title('Mean recovery window duration (d)', fontsize=7)

tmp = np.corrcoef(df_rate_onset_relThresh_clim, np.log10(depth))[0,1]
axes[0,0].annotate('$r={:.2f}$'.format(tmp), xy=(0.05, 0.85), xycoords='axes fraction')
tmp = np.corrcoef(df_rate_decline_relThresh_clim, np.log10(depth))[0,1]
axes[0,1].annotate('$r={:.2f}$'.format(tmp), xy=(0.65, 0.85), xycoords='axes fraction')
tmp = np.corrcoef(df_reaction_window_relThresh_duration_clim, np.log10(depth))[0,1]
axes[0,2].annotate('$r={:.2f}$'.format(tmp), xy=(0.05, 0.85), xycoords='axes fraction')
tmp = np.corrcoef(df_coping_window_relThresh_duration_clim, np.log10(depth))[0,1]
axes[1,0].annotate('$r={:.2f}$'.format(tmp), xy=(0.05, 0.85), xycoords='axes fraction')
tmp = np.corrcoef(df_recovery_window_duration_clim, np.log10(depth))[0,1]
axes[1,1].annotate('$r={:.2f}$'.format(tmp), xy=(0.65, 0.85), xycoords='axes fraction')

auto_label(axes.flatten()[:-1], x_offset=-0.1, y_offset=0.05)

# %% [markdown]
# ### SF10 
# count/duration/intensity_mean trend

# %%
fig = plt.figure(figsize=(18/2.54, 16/2.54), dpi=500)
plt.subplots_adjust(wspace=0.2, hspace=0.2)
ax2 = plt.subplot(322, projection=proj)
ax4 = plt.subplot(324, projection=proj)
ax6 = plt.subplot(326, projection=proj)

lw = 0.0
plot_basic(ax=ax2, var=df_slope['count'], loc_linewidth=lw, loc_vmin=-1, loc_vmax=1, loc_s=2, loc_cbar_on='vertical',\
            loc_cbar_title='Count trend (10a$^{-1}$)')

plot_basic(ax=ax4, var=df_slope['intensity_mean'], loc_linewidth=lw, loc_vmin=-0.3, loc_vmax=0.3, loc_s=2, \
           loc_cbar_on='vertical', loc_cbar_title='Intensity mean trend (\u00B0C 10a$^{-1}$)')

plot_basic(ax=ax6, var=df_slope['duration'], loc_linewidth=lw, loc_vmin=-3, loc_vmax=3, loc_s=2, loc_cbar_on='vertical', \
           loc_cbar_title='Duration trend (d 10a$^{-1}$)')

ax1 = add_equal_axes(ax2, 'left', 0.05)
tmpdf = df_count_mean
tmpmk = mk.original_test(tmpdf)
ax1.plot(np.arange(1979,2023), tmpdf, linewidth=1, c='#5770db')
ax1.plot(np.arange(1979,2023), tmpmk[7]*np.arange(0,2023-1979)+tmpmk[8], alpha=0.45, linewidth=2.2, c='#5770db')
ax1.set_ylabel('Count',)
ax1.annotate(
    '{:+.2f} '.format(tmpmk[7]*10) + '10a$^{-1},\ p$ < 0.005',
    xy=(0.45, 0.05), xycoords='axes fraction',
    color='k'
    )

ax3 = add_equal_axes(ax4, 'left', 0.05)
tmpdf = df_intensity_mean_mean
tmpmk = mk.original_test(tmpdf)
ax3.plot(np.arange(1979,2023), tmpdf, linewidth=1, c='#5770db')
ax3.plot(np.arange(1979,2023), tmpmk[7]*np.arange(0,2023-1979)+tmpmk[8], alpha=0.45, linewidth=2.2, c='#5770db')
ax3.set_ylabel('Intensity mean (\u00B0C)',)
ax3.annotate(
    '{:+.2f} '.format(tmpmk[7]*10) + '\u00B0C 10a$^{-1},\ p$ < 0.005',
    xy=(0.45, 0.05), xycoords='axes fraction',
    color='k'
    )

ax5 = add_equal_axes(ax6, 'left', 0.05)
tmpdf = df_duration_mean
tmpmk = mk.original_test(tmpdf)
ax5.plot(np.arange(1979,2023), tmpdf, linewidth=1, c='#5770db')
ax5.plot(np.arange(1979,2023), tmpmk[7]*np.arange(0,2023-1979)+tmpmk[8], alpha=0.45, linewidth=2.2, c='#5770db')
ax5.set_ylabel('Duration (d)',)
ax5.annotate(
    '{:+.2f} '.format(tmpmk[7]*10) + 'd 10a$^{-1},\ p$ < 0.005',
    xy=(0.06, 0.85), xycoords='axes fraction',
    color='k'
    )

bbox = ax1.get_position()
width = bbox.x1 - bbox.x0
ax0 = add_equal_axes([ax1, ax2], 'top', pad=0.07, v_width=width, ha='center')
tmpdf = number_of_lakes_with_lhw 
tmpmk = mk.original_test(tmpdf)
print(tmpmk[2], tmpmk[7])
print(tmpdf.iloc[0], tmpdf.iloc[-1])
ax0.plot(np.arange(1979,2023), tmpdf, linewidth=1, c='#5770db')
ax0.plot(np.arange(1979,2023), tmpmk[7]*np.arange(0,2023-1979)+tmpmk[8], alpha=0.45, linewidth=2.2, c='#5770db')
ax0.set_ylabel('Number of lakes')
ax0.annotate(
    '{:+.2f} '.format(tmpmk[7]*10) + '10a$^{-1},\ p$ < 0.005',
    xy=(0.45, 0.05), xycoords='axes fraction',
    color='k'
    )

auto_label(ax0, ax1, ax2, ax3, ax4, ax5, ax6)

# %% [markdown]
# ### SF12
# Seasonal RMSE

# %%
# Found the solution at https://stackoverflow.com/questions/56838187/how-to-create-spacing-between-same-subgroup-in-seaborn-boxplot
from matplotlib.patches import PathPatch
def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])

        # set black edge
        box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
        if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax2.artists
            box_patches = ax.artists
        num_patches = len(box_patches)
        lines_per_boxplot = len(ax.lines) // num_patches
        for i, patch in enumerate(box_patches):
            # Set the linecolor on the patch to the facecolor, and set the facecolor to None
            patch.set_edgecolor('k')

            # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
            # Loop over them here, and use the same color as above
            for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
                line.set_color('k')
                line.set_mfc('k')  # facecolor of fliers
                line.set_mec('k')  # edgecolor of fliers

        # Also fix the legend
        for legpatch in ax.legend_.get_patches():
            legpatch.set_edgecolor('k')

# %%
fig, axes = plt.subplots(2, 2, figsize=(16/2.54, 14/2.54), dpi=500, sharex=True, sharey=True)
sns.boxplot(ax=axes[0,0], data=bx_sh, x='season2', y='y', hue='hue', palette=['#e872a5','#f2b1cd'], saturation=1, linewidth=0.8, fliersize=2)
sns.boxplot(ax=axes[1,0], data=bx_lh, x='season2', y='y', hue='hue', palette=['#b06bc4','#d7b5e1'], saturation=1, linewidth=0.8, fliersize=2)
sns.boxplot(ax=axes[0,1], data=bx_netlw, x='season2', y='y', hue='hue', palette=['#26c3eb','#8fe4f6'], saturation=1, linewidth=0.8, fliersize=2)
sns.boxplot(ax=axes[1,1], data=bx_mlsw, x='season2', y='y', hue='hue', palette=['#5862bd','#abb0de'], saturation=1, linewidth=0.8, fliersize=2)

axes[0,0].set_title('Sensible heat flux')
axes[1,0].set_title('Latent heat flux')
axes[0,1].set_title('Longwave radiation')
axes[1,1].set_title('Shortwave radiation')

plt.subplots_adjust(hspace=0.3)

for ax in axes.flatten():
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, frameon=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(['DJF', 'MAM', 'JJA', 'SON'])
    ax.yaxis.set_tick_params(labelleft=True)
    ax.xaxis.set_tick_params(labelbottom=True)

axes[0,0].set_ylabel('RMSE (\u00B0C)')
axes[1,0].set_ylabel('RMSE (\u00B0C)')

adjust_box_widths(fig, 0.9)
auto_label(axes[0,0], axes[0,1], axes[1,0], axes[1,1])

# %% [markdown]
# ### SF14
# detrended

# %%
def plot_unnamed(ax, ivar, flag, lw_ratio=1):
    x_future = np.arange(2023, 2101)
    if flag == 'detrended':
        mme = hw_future_detrended_mme_global.sel(var=ivar, time=slice('2023', '2100'), scenario='ssp585')
        ax.plot(x_future, mme, lw=1.5*lw_ratio, c=ssp_colors[2], ls='-')
        # ax.plot(x_future, mme, lw=1.5*lw_ratio, c=ssp_colors[2], ls='dashdot')
        std = hw_future_detrended_std_global.sel(var=ivar, time=slice('2023', '2100'), scenario='ssp585')
        low_bound = mme - std
        upp_bound = mme + std 
        ax.fill_between(x_future, low_bound, upp_bound, alpha=0.2, lw=0, color=ssp_colors[2])
    return ax

# %%
fig, axes = plt.subplots(2, 3, figsize=(16/2.54, 8/2.54))
for ax in axes.flatten():
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(width=0.5, length=2)
plt.subplots_adjust(wspace=0.3, hspace=0.6)

plot_unnamed(axes.flatten()[0], 'rate_onset_relThresh', 'detrended', 0.6)
axes.flatten()[0].set_title('Rate of onset (\u00B0C d$^{-1}$)', fontsize=7)

plot_unnamed(axes.flatten()[1], 'rate_decline_relThresh', 'detrended', 0.6)
axes.flatten()[1].set_title('Rate of decline (\u00B0C d$^{-1}$)', fontsize=7)

plot_unnamed(axes.flatten()[2], 'reaction_window_relThresh_duration', 'detrended', 0.6)
axes.flatten()[2].set_title('Reaction window duration (d)', fontsize=7)

plot_unnamed(axes.flatten()[3], 'coping_window_relThresh_duration', 'detrended', 0.6)
axes.flatten()[3].set_title('Coping window duration (d)', fontsize=7)

plot_unnamed(axes.flatten()[4], 'recovery_window_duration', 'detrended', 0.6)
axes.flatten()[4].set_title('Recovery window duration (d)', fontsize=7)

axes.flatten()[5].set_axis_off()

line1 = Line2D([0], [0], label='Using a fixed baseline\nwith the detrended data', color=ssp_colors[2], lw=1.5*0.6, ls='dashdot')

auto_label(np.delete(axes.flatten(), [5]), x_offset=-0.1, y_offset=0.05)

# %% [markdown]
# ### SF5
# LHW properties validation (spatial)

# %%
fig = plt.figure(figsize=(18/2.54, 20/2.54), dpi=500)
plt.subplots_adjust(wspace=0.4, hspace=0.3)
ax1 = plt.subplot(421, projection=proj)
ax2 = plt.subplot(422, projection=proj)
ax3 = plt.subplot(423, projection=proj)
ax4 = plt.subplot(424, projection=proj)
ax5 = plt.subplot(425, projection=proj)

im = plot_basic(var_subset=mae_hw_clim['count'], lat_subset=lats[tested_lakes], lon_subset=lons[tested_lakes], \
                loc_vmin=0, loc_vmax=3, ax=ax1, loc_linewidth=0.0, loc_cmap_name='viridis_r', loc_title='Count',\
                      loc_cbar_on='vertical', loc_cbar_title='MAE', loc_cbar_title_loc='top')

plot_basic(var_subset=mae_hw_clim['duration'], lat_subset=lats[tested_lakes], lon_subset=lons[tested_lakes], \
           loc_vmin=0, loc_vmax=1, ax=ax2, loc_linewidth=0.0, loc_cmap_name='viridis_r', loc_title='Duration (d)', \
            loc_cbar_on='vertical', loc_cbar_title='MAE', loc_cbar_title_loc='top')

plot_basic(var_subset=mae_hw_clim['intensity_mean'], lat_subset=lats[tested_lakes], lon_subset=lons[tested_lakes], \
           loc_vmin=0, loc_vmax=1, ax=ax3, loc_linewidth=0.0, loc_cmap_name='viridis_r', loc_title='Intensity mean (\u00B0C)',\
              loc_cbar_on='vertical', loc_cbar_title='MAE', loc_cbar_title_loc='top')

plot_basic(var_subset=mae_hw_clim['rate_onset_relThresh'], lat_subset=lats[tested_lakes], lon_subset=lons[tested_lakes], \
           loc_vmin=0, loc_vmax=1, ax=ax4, loc_linewidth=0.0, loc_cmap_name='viridis_r', \
            loc_title='Rate of onset (\u00B0C d$^{-1}$)', loc_cbar_on='vertical', loc_cbar_title='MAE', loc_cbar_title_loc='top')

plot_basic(var_subset=mae_hw_clim['rate_decline_relThresh'], lat_subset=lats[tested_lakes], lon_subset=lons[tested_lakes], \
           loc_vmin=0, loc_vmax=1, ax=ax5, loc_linewidth=0.0, loc_cmap_name='viridis_r', \
            loc_title='Rate of decline (\u00B0C d$^{-1}$)', loc_cbar_on='vertical', loc_cbar_title='MAE', loc_cbar_title_loc='top')

auto_label(ax1, ax2, ax3, ax4, ax5)

# %% [markdown]
# ### SF6
# LHW properties validation (temporal)

# %%
fig = plt.figure(figsize=(18/2.54, 20/2.54), dpi=500)
plt.subplots_adjust(wspace=0.3, hspace=0.3)

axes = []
axes.append(plt.subplot(421))
axes.append(plt.subplot(422))
axes.append(plt.subplot(423))
axes.append(plt.subplot(424))
axes.append(plt.subplot(425))

axes = np.array(axes)

for ivar, var in enumerate(['count','duration','intensity_mean','rate_onset_relThresh','rate_decline_relThresh']):
    ax = axes[ivar]
    ax.plot(np.arange(2000, 2021, 1), hw_satt.mean(dim='id').sel(var=var), lw=1., c='tab:blue', label='Lakes_cci')
    ax.plot(np.arange(2000, 2021, 1), hw_sim.mean(dim='id').sel(var=var), lw=1., c='tab:green', label='CSFLake')
    tmpmk1 = mk.original_test(hw_satt.mean(dim='id').sel(var=var))
    tmpmk2 = mk.original_test(hw_sim.mean(dim='id').sel(var=var))
    ax.plot(np.arange(2000, 2021, 1), np.arange(0, 2021-2000, 1)*tmpmk1[7] + tmpmk1[8], c='tab:blue', alpha=0.45, lw=2.2)
    ax.plot(np.arange(2000, 2021, 1), np.arange(0, 2021-2000, 1)*tmpmk2[7] + tmpmk2[8], c='tab:green', alpha=0.45, lw=2.2)
    ax.set_xlabel('Year')

axes[0].set_ylabel('Count')
axes[1].set_ylabel('Duration (d)')
axes[2].set_ylabel('Intensity mean (\u00B0C)')
axes[3].set_ylabel('Rate of onset (\u00B0C d$^{-1}$)')
axes[4].set_ylabel('Rate of decline (\u00B0C d$^{-1}$)')

axes[4].legend(frameon=False, bbox_to_anchor=(1.08, 0.5), ncol=1, loc='center left', labelspacing=1.2, )

auto_label(axes)

# %% [markdown]
# ### SF7
# IDF plot

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16/2.54, 6/2.54), sharey=True, sharex=True)
plt.subplots_adjust(wspace=0.45)  # put this before create cbax!
cbax2 = add_equal_axes(ax2, 'right', 0.01, h_width=0.02)
cbax1 = add_equal_axes(ax1, 'right', 0.01, h_width=0.02)
cbax1.set_title('Count', fontsize=7)
cbax2.set_title('Count', fontsize=7)
sns.histplot(stack_satt, ax=ax1, x='duration', y='intensity_mean', binwidth=[1, 0.1], stat='count', discrete=(True, False), \
             cbar=True, cbar_ax=cbax1, vmin=None, vmax=None, norm=matplotlib.colors.LogNorm(), cmap='light:#045993')

sns.histplot(stack_sim, ax=ax2, x='duration', y='intensity_mean', binwidth=[1, 0.1], stat='count', discrete=(True, False), \
             cbar=True, cbar_ax=cbax2, vmin=None, vmax=None, norm=matplotlib.colors.LogNorm(), cmap='light:#118011')
ax1.set_xlabel('Duration (d)')
ax2.set_xlabel('Duration (d)')
ax1.set_ylabel('Intensity mean (\u00B0C)')
ax2.set_ylabel('Intensity mean (\u00B0C)', visible=True)
ax2.tick_params(labelleft=True)
ax1.set_title('Lakes_cci', fontsize=7)
ax2.set_title('CSFLake', fontsize=7)
auto_label(ax1, ax2)

# %% [markdown]
# ### SF11
# Recovery window between years

# %%
fig = plt.figure(figsize=(12/2.54, 12/2.54), dpi=500)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
ax2 = plt.subplot(211, projection=proj)
ax4 = plt.subplot(212, projection=proj)

lw = 0.
plot_basic(df_slope['recovery_window_between_years'], ax=ax2, loc_vmin=-20, loc_vmax=20, loc_cbar_on='vertical', loc_linewidth=lw)

ax1 = add_equal_axes(ax2, 'left', pad=0.08, h_width=0.5)
tmpdf = df_recovery_window_between_years.mean(axis=1)
tmpmk = mk.original_test(tmpdf)
print(tmpmk[2], tmpmk[7])
ax1.plot(np.arange(1979,2023), tmpdf, linewidth=1, c='k')
ax1.plot(np.arange(1979,2023), tmpmk[7]*np.arange(0,2023-1979)+tmpmk[8], alpha=0.45, linewidth=2.2, c='k')
ax1.set_ylabel('Reaction window between years (d)',)
ax1.annotate(
    '{:.2f} '.format(tmpmk[7]*10) + 'd 10a$^{-1},\ p$ < 0.005',
    xy=(0.05, 0.05), xycoords='axes fraction',
    color='k'
    )

plot_basic(var_subset=hw_future_clim_diff.sel(var='recovery_window_between_years', scenario='ssp585'), \
           lat_subset=lats[include], lon_subset=lons[include], ax=ax4, loc_vmax=100, loc_vmin=-600, \
            loc_vcenter=0, loc_norm_on=True, loc_cbar_on='vertical')

ax3 = add_equal_axes(ax4, 'left', pad=0.08, h_width=0.5)
plot_scenario(ax3, 'recovery_window_between_years', lw_ratio=0.6, legend_on=True)
ax3.set_ylabel('Reaction window between years (d)',)

ax2.set_title('1979\u20132022 change rates (d 10a$^{-1}$)', fontsize=7)
ax4.set_title('2071\u20132100 mean minus 1979\u20132008 mean under SSP5-8.5 (d)', fontsize=7)

auto_label(ax1, ax2, ax3, ax4)

# %% [markdown]
# ### SF13
# Future IDF plot

# %%
fig, axes = plt.subplots(2, 3, figsize=(18/2.54, 11/2.54), sharey=True, sharex=True)

plt.subplots_adjust(wspace=0.25, hspace=0.4)  # put this before creating cbax

axes.flatten()[-1].axis('off')

sns.histplot(stack_ctl, ax=axes[0,0], x='duration', y='intensity_mean', binwidth=[1, 0.1], stat='count', discrete=(True, False), \
             cbar=False, vmin=None, vmax=None, norm=matplotlib.colors.LogNorm(), cmap='Spectral_r')
sns.histplot(stack_gfdl, ax=axes[0,1], x='duration', y='intensity_mean', binwidth=[1, 0.1], stat='count', discrete=(True, False), \
             cbar=False, vmin=None, vmax=None, norm=matplotlib.colors.LogNorm(), cmap='Spectral_r')
sns.histplot(stack_ipsl, ax=axes[0,2], x='duration', y='intensity_mean', binwidth=[1, 0.1], stat='count', discrete=(True, False), \
             cbar=False, vmin=None, vmax=None, norm=matplotlib.colors.LogNorm(), cmap='Spectral_r')
sns.histplot(stack_mri, ax=axes[1,0], x='duration', y='intensity_mean', binwidth=[1, 0.1], stat='count', discrete=(True, False), \
             cbar=False, vmin=None, vmax=None, norm=matplotlib.colors.LogNorm(), cmap='Spectral_r')

cbax = add_equal_axes(axes[1,1], 'right', 0.035, h_width=0.016)
cbax.set_title('Count', fontsize=7)
sns.histplot(stack_ukesm1, ax=axes[1,1], x='duration', y='intensity_mean', binwidth=[1, 0.1], stat='count', discrete=(True, False), \
             cbar=True, cbar_ax=cbax, vmin=None, vmax=None, norm=matplotlib.colors.LogNorm(), cmap='Spectral_r')

for ax in axes.flatten()[:-1]:
    ax.tick_params(labelleft=True, labelbottom=True)
    ax.set_ylabel('Intensity mean (\u00B0C)')
    ax.set_xlabel('Duration (d)')

axes[0,0].set_title('ERA5-Land', fontsize=7)
axes[0,1].set_title('GFDL-ESM4', fontsize=7)
axes[0,2].set_title('IPSL-CM6A-LR', fontsize=7)
axes[1,0].set_title('MRI-ESM2-0', fontsize=7)
axes[1,1].set_title('UKESM1-0-LL', fontsize=7)

auto_label(axes.flatten()[:-1])


# %%
fig, ax = plt.subplots(figsize=(12/2.54, 9/2.54))
plot_scenario(ax, 'intensity_max_relThresh', lw_ratio=0.6, legend_on=True)
ax.set_xlabel('Year')
ax.set_ylabel('Maximum intensity relative to the 90th percentile threshold (\u00B0C)')


