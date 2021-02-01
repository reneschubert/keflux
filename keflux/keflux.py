# load necessary modules
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# additionally necessary for variant 2
import pyproj
from scipy import interpolate
from scipy.signal import convolve2d

def comp_keflux(u,v,lon_x,lat_y,scales,variant,gs):
    # u          zonal velocity component (needs to exist on the same grid as v)
    # v          meridional velocity component  (needs to exist on the same grid as u)
    # lon_x      longitudes or zonal metric distances of u and v
    # lat_y      latitudes or meridional metric distances of u and v
    # scales     a vector of the scales in kilometer, the scale kinetic energy flux shall be computed for
    # variant    'geographic' or 'metric'
    #            - 'geographic': compute the scale kinetic energy flux on the grid, u and v exist on
    #            - 'metric': project and interpolate u and v first onto a regular metric grid
    # gs         grid spacing in kilometer (assumes quadratic cells!)
    #            - 'geographic': 2D-array
    #            - 'metric': scalar (grid-spacing of the regular grid, the data shall be interpolated on)
    
   
    if variant == 'geographic':
        pi = np.zeros((len(scales),u.shape[0],u.shape[1])) + np.nan
        counter = 0
        for scale_tmp in scales:
            # create top-hat convolution kernel for each grid-cell (with a diameter that equals the respective scale)
            radius = int( (scale_tmp / gs.min().compute() / 2 ).round()) # get the radius in grid-cells that covers the convolution kernel also for the smallest grid-spacing  
            window_size = 2 * radius + 1
            gsr = gs.rolling(x=window_size, center=True).construct("lon_window").rolling(y=window_size, center=True).construct("lat_window")
            gsr_lat = gsr.cumsum("lat_window")
            gsr_lat -= gsr_lat.isel(lat_window=radius)
            gsr_lon = gsr.cumsum("lon_window")
            gsr_lon -= gsr_lon.isel(lon_window=radius)
            circ = ((gsr_lat ** 2 + gsr_lon ** 2) ** 0.5 < scale_tmp / 2)
            Asum = (circ * (gsr ** 2)).sum(dim = ["lat_window","lon_window"])
            
            # convolute u
            uA  = gs ** 2 * u # multiplication with area
            uAr = uA.rolling(x=window_size, center=True).construct("lon_window").rolling(y=window_size, center=True).construct("lat_window")
            um = ((uAr * circ).sum(dim = ["lat_window","lon_window"]) / Asum)

            # convolute v
            vA  = gs ** 2 * v # multiplication with area
            vAr = vA.rolling(x=window_size, center=True).construct("lon_window").rolling(y=window_size, center=True).construct("lat_window")
            vm = ((vAr * circ).sum(dim = ["lat_window","lon_window"]) / Asum)

            # convolute u²
            uuA  = gs ** 2 * u * u # multiplication with area
            uuAr = uuA.rolling(x=window_size, center=True).construct("lon_window").rolling(y=window_size, center=True).construct("lat_window")
            uum = ((uuAr * circ).sum(dim = ["lat_window","lon_window"]) / Asum)

            # convolute v²
            vvA  = gs ** 2 * v * v # multiplication with area
            vvAr = vvA.rolling(x=window_size, center=True).construct("lon_window").rolling(y=window_size, center=True).construct("lat_window")
            vvm = ((vvAr * circ).sum(dim = ["lat_window","lon_window"]) / Asum)

            # convolute uv
            uvA  = gs ** 2 * u * v # multiplication with area
            uvAr = uvA.rolling(x=window_size, center=True).construct("lon_window").rolling(y=window_size, center=True).construct("lat_window")
            uvm = ((uvAr * circ).sum(dim = ["lat_window","lon_window"]) / Asum)


            # compute the horizontal derivatives of um and vm (using centred difference)
            dumdx = (um.shift(x=-1) - um.shift(x=1)) / ((gs + gs.shift(x=-1) / 2 + gs.shift(x=1) / 2) * 1000) # grid spacing converted into m
            dumdy = (um.shift(y=-1) - um.shift(y=1)) / ((gs + gs.shift(y=-1) / 2 + gs.shift(y=1) / 2) * 1000)
            dvmdx = (vm.shift(x=-1) - vm.shift(x=1)) / ((gs + gs.shift(x=-1) / 2 + gs.shift(x=1) / 2) * 1000)
            dvmdy = (vm.shift(y=-1) - vm.shift(y=1)) / ((gs + gs.shift(y=-1) / 2 + gs.shift(y=1) / 2) * 1000)

            # compute the scale kinetic energy flux
            rho_0 = 1024 # define standard density in kg/m³
            pi_tmp = -1 * rho_0 * ((uum - um ** 2) * dumdx + (uvm - um * vm) * (dumdy + dvmdy) + (vvm - vm ** 2) * dvmdy)
            # set the pixels at the boundary to NaN, where the convolution kernel extends over the boundary
            pi_tmp2 = np.zeros((pi_tmp.shape)) + np.nan
            pi_tmp2[radius:-radius,radius:-radius] = pi_tmp[radius:-radius,radius:-radius]
             
            pi[counter,:,:] = pi_tmp2
            counter+=1
            
        return pi,lon_x,lat_y
    
    if variant == 'metric':
        # project and interpolate u and v onto a regular metric grid with a grid spacing of gs [km]
        a = pyproj.Transformer.from_crs(4326,3395).transform(lon_x,lat_y) # project WGS84 onto metric grid
        y3 = a[0]
        x3 = a[1]
        x3min = np.nanmin(x3,1)
        y3min = np.nanmin(y3,0)
        Y3min,X3min = np.meshgrid(y3min, x3min)
        x1=(x3-X3min)/1000
        y1=(y3-Y3min)/1000
        x_len = int(np.floor(np.amax(x1))+1)
        y_len = int(np.floor(np.amax(y1))+1)
        x_dim = np.linspace(0, x_len, int(x_len/gs), endpoint=False)
        y_dim = np.linspace(0, y_len, int(y_len/gs), endpoint=False)
        x2, y2 = np.meshgrid(x_dim, y_dim)
        ui = interpolate.griddata((x1.ravel(), y1.ravel()), np.array(u).ravel(), (x2, y2), method='linear')
        vi = interpolate.griddata((x1.ravel(), y1.ravel()), np.array(v).ravel(), (x2, y2), method='linear')
        
        pi = np.zeros((len(scales),len(y_dim)-2,len(x_dim)-2)) + np.nan
        counter = 0
        for scale_tmp in scales:
            # create convolution kernel (which is here the same for every grid-cell)
            radius = int(scale_tmp/2)
            y,x = np.ogrid[-radius: radius+1, -radius: radius+1]
            disk = x**2+y**2 <= radius**2
            disk = disk.astype(float)
            disk = disk/sum(sum(disk))
            # convolute the fields with the kernel
            um = convolve2d(ui,disk, mode='same', boundary='fill', fillvalue=0)
            uum = convolve2d(ui**2,disk, mode='same', boundary='fill', fillvalue=0)
            vm = convolve2d(vi,disk, mode='same', boundary='fill', fillvalue=0)
            vvm = convolve2d(vi**2,disk, mode='same', boundary='fill', fillvalue=0)
            uvm = convolve2d(ui*vi,disk, mode='same', boundary='fill', fillvalue=0)
            # compute the horizontal derivatives of um and vm (using centred difference)
            dumdx = (um[1:-1,2:]-um[1:-1,:-2])/(gs*2*1000)
            dumdy = (um[2:,1:-1]-um[:-2,1:-1])/(gs*2*1000)
            dvmdx = (vm[1:-1,2:]-vm[1:-1,:-2])/(gs*2*1000)
            dvmdy = (vm[2:,1:-1]-vm[:-2,1:-1])/(gs*2*1000)
            # compute the scale kinetic energy flux
            rho_0 = 1024 # define standard density in kg/m³
            pi_tmp = -1 * rho_0 * ((uum[1:-1,1:-1] - um[1:-1,1:-1] ** 2) * dumdx + (uvm[1:-1,1:-1] - um[1:-1,1:-1] * vm[1:-1,1:-1]) * (dumdy + dvmdy) + (vvm[1:-1,1:-1] - vm[1:-1,1:-1] ** 2) * dvmdy)  
            # set the pixels at the boundary to NaN, where the convolution kernel extends over the boundary
            pi_tmp2 = np.zeros((pi_tmp.shape)) + np.nan
            pi_tmp2[radius:-radius,radius:-radius] = pi_tmp[radius:-radius,radius:-radius]
             
            pi[counter,:,:] = pi_tmp2
            counter+=1
        return pi,np.arange(0,len(x_dim)-2)*gs,np.arange(0,len(y_dim)-2)*gs
