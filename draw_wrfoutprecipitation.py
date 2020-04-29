from netCDF4 import Dataset
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.cm import get_cmap
import matplotlib.colors as colors
from cartopy.io.shapereader import Reader
import cartopy.feature as cfe
import cmaps
from matplotlib.backends.backend_pdf import PdfPages
from wrf import (get_cartopy, latlon_coords, to_np, cartopy_xlim, cartopy_ylim,
                 getvar, ALL_TIMES)
                 
f = Dataset('wrfout_d01')
# rainc = np.array(f.variables['RAINC'][:])
# rainnc = np.array(f.variables['RAINNC'][:])

rainc = getvar(f, 'RAINC', timeidx=ALL_TIMES)
rainnc = getvar(f, 'RAINNC', timeidx=ALL_TIMES)

# f = xr.open_dataset('wrfout_d01')
# rainc = f['RAINC']
# rainnc = f['RAINNC']

tot = rainc.values + rainnc.values
wrf_time = f.variables['Times']

# tot = rainc + rainnc
dim_tot = tot.shape
h1prep = tot[2:dim_tot[0]:2,:,:] - tot[0:dim_tot[0]-2:2,:,:]

# lats = f['XLAT'].values[0]
# lons = f['XLONG'].values[0]

lats = f.variables['XLAT'][0]
lons = f.variables['XLONG'][0]

# lats, lons = latlon_coords(rainc)
cart_proj = get_cartopy(rainc)
# print(cart_proj)

levels = [0.1,0.5,1.,2.,3.,4.,5.,6.,8.,10.,20.,40]
norm = colors.BoundaryNorm(boundaries=np.array(levels), ncolors=len(levels)-1)

shp_path = '/Users/james/Documents/GitHub/py_china_shp/'
reader = Reader(shp_path  +  'Province_9/Province_9.shp')
provinces = cfe.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='k', facecolor='none')


with PdfPages('multipage_pdf.pdf') as pdf:
    for i in range(3):
        fig = plt.figure(figsize=(14,8.5))
        ax = fig.add_subplot(111, projection=cart_proj)
        print(ax)   

        ax.add_feature(provinces, linewidth=0.6)
        ax.coastlines('50m', linewidth=0.8)
    
        pcm = ax.contourf(lons, lats, h1prep[i], levels, norm=norm,
                     transform=ccrs.PlateCarree(), extend='both',
                     cmap=cmaps.cmorph[1:])
        #pcm.cmap.set_over('white')

        pcm.cmap.set_under(cmaps.cmorph.colors[0])  

        # Set the map bounds
        # 两种方式设定画图范围
        # 方式1
        # xs, ys, _ = cart_proj.transform_points(ccrs.PlateCarree(),np.array([lons[0,0],lons[-1,-1]]),np.array([lats[0,0],lats[-1,-1]])).T
        # xlimits = xs.tolist()
        # ylimits = ys.tolist()
        # ax.set_xlim(xlimits)
        # ax.set_ylim(ylimits)
        # print(xlimits)
        # print(ylimits)
        # 方式2
        # ax.set_xlim(cartopy_xlim(rainc))
        # ax.set_ylim(cartopy_ylim(rainc))  
        lon_formatter = cticker.LongitudeFormatter()
        lat_formatter = cticker.LatitudeFormatter()
        ax.set_title("WRF Prepcipitation {}".format(str(wrf_time[i],'utf-8')))
        ax.gridlines(linewidth=0.5, color='gray', linestyle='--')   

        # plt.colorbar(pcm, ax=ax, ticks=levels, extendfrac='auto', 
        #              aspect=20, extendrect=True, format='%3.1f', drawedges=True, shrink=.7)
        plt.colorbar(pcm, ax=ax, ticks=levels, extendfrac='auto', 
                     aspect=18, format='%3.1f', shrink=.95)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

    # plt.suptitle('Figure title')
    # # shrink 调整colorbar大小   

    # plt.show()