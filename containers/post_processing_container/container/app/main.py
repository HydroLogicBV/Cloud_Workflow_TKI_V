import numpy as np
import matplotlib.pyplot as plt
import os
import netCDF4 as nc
import random
import contextily as cx
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%%
class template:
    """ class to make a template for the simulation output. Stores useful information like grid size, cell locations etc. """
    def __init__(self, data_folder):
        found_template = False
        """Lees een template uit, dit zorgt er voor dat je niet zoveel data hoeft op te slaan """
        output_folder_list = os.listdir(data_folder)
        for folder in output_folder_list: 
            files = os.listdir(os.path.join(data_folder, folder))
            map_file = [file for file in files if file.endswith("map.nc")]
            if len(map_file) != 1: # if this folder does not have exactly 1 map file
                continue
            else: # hij breakt na deze dus dan stop de for loop waar hij in zat automatisch
                print("Using folder {} as a template for the 2D grid generation".format(folder))
                found_template = True
                map_file = map_file[0]
                ds_map = nc.Dataset(os.path.join(data_folder, folder, map_file))
                self.x_face = ds_map.variables['mesh2d_face_x'][:]
                self.y_face = ds_map.variables['mesh2d_face_y'][:]
                ds_map.close()
                self.x_array, self.y_array, self.x_coor_dict, self.y_coor_dict, self.cell_size = template.setup_2d_grid(self.x_face, self.y_face)
                self.rows, self.cols = self.get_row_col_for_grid()
                break 
        if found_template == False:
            raise Except("Did not find a template!")
        
        
    def setup_2d_grid(x_face, y_face):
        """ Maak een 2d grid van je 1D waterdieptes. Dit maakt een template daarvoor  """
        # to find the cellsize, calculate distance between two cells in x and y direction. If these are the same, cell size is found
        i, found_cell_size, prev_distances = 0, False, []
        while found_cell_size == False: # first, find the cell size. Do this by subtractting x_diff and y_diff 
            dx = abs(x_face[i+1] - x_face[i])
            if len(prev_distances) == 0:
                prev_distances.append(dx)
            if len(prev_distances) < 5:
                if dx == prev_distances[-1]:
                    prev_distances.append(dx)
                else:
                    prev_distances = []
            else:
                found_cell_size = True
                cell_size = prev_distances[-1]
                print("Cell size is {} meters".format(cell_size))
            if i > len(x_face)-1:
                raise Exception("ERROR: Could not find the  cell size")
           
        # another check
        dx_check = (max(x_face) - min(x_face)) / (len(list(set(x_face)) ) - 1)
        dy_check = (max(y_face) - min(y_face)) / (len(list(set(y_face)) ) - 1)
        if dx_check != dy_check and dy_check != cell_size:
            raise Exception("Check in grid cell size did not succeed. Calculated cell size is {}, dx_check = {}, dy_check = {}".format(cell_size, dx_check, dy_check))
        
        # find dimension of grid
        n = round(( max(x_face) - min(x_face) ) / cell_size + 1,4)
        m = round(( max(y_face) - min(y_face) ) / cell_size + 1,4)
        if not( m.is_integer() and n.is_integer()):
            raise Exception("estimate of amount of rows and columns are not exactly integers")
        else:
            m = int(m)
            n = int(n)
            print("Amount of rows is {}, amount of columns is {}. This makes total of {} cells, where the amount of cells with data is {}".format(m, n, n*m, len(x_face)))
        
        # initiate grid
        x_array = np.arange(min(x_face), cell_size * n + min(x_face) + 0.01, cell_size)
        y_array = np.arange(min(y_face), cell_size * m + min(y_face) + 0.01, cell_size)
        
        # make dict to be able to look up index from coordinates
        x_coor_dict, y_coor_dict = {}, {}
        for i in range(len(x_array)):
            x_coor_dict[x_array[i]] = i
        for i in range(len(y_array)):
            y_coor_dict[y_array[i]] = i 
        return [x_array, y_array, x_coor_dict, y_coor_dict, cell_size]
    
    def get_row_col_for_grid(self):
        """ Dit is handig voor het indexen van elk elmeent uit je 1D waterdieptes"""
        rows = []
        cols = []
        for i in range(len(self.x_face)):
            rows.append(self.y_coor_dict[self.y_face[i]])
            cols.append(self.x_coor_dict[self.x_face[i]])
        return [rows, cols]
    

class model_data:
    """ class for saving all the useful data from the simulation output"""
    def __init__(self, data_folder):
        self.waterdepths = model_data.read_nc_files(data_folder)
        self.template = template(data_folder)
        
        self.max_waterdepths = {}
        for key in self.waterdepths.keys():
            self.max_waterdepths[key] = np.amax(self.waterdepths[key], axis = 0)
  
    def read_nc_files(data_folder): 
        """ lees van elke folder de map.nc file uit. haal de waterdiepte er uit.
        Sla deze op als rijen van waterdiepte per tijdstap."""
        output_folder_list = os.listdir(data_folder)
        water_depth_dict = {}
        shapes = {}
        for folder in output_folder_list: 
            files = os.listdir(os.path.join(data_folder, folder))
            map_file = [file for file in files if file.endswith("map.nc")]
            if len(map_file) == 0:
                print("Folder {} has no map file in it! Skipping this folder when plottin data".format(folder))  
            elif len(map_file) > 1:
                print("Folder {} has more than 1 map file in it! Skipping this folder when plottin data".format(folder))
            elif len(map_file) == 1:
                map_file = map_file[0]
                ds_map = nc.Dataset(os.path.join(data_folder, folder, map_file))
                water_depth = np.array(ds_map.variables['mesh2d_waterdepth'][:])
                ds_map.close()
                water_depth_dict[folder] = water_depth
                shapes[folder] = water_depth.shape
        unique_shapes = set(shapes.values())
        if len(unique_shapes) != 1:
            print("Not all waterdepth shapes are the same! Here is a list of the shapes:")
            print(shapes)
        return water_depth_dict
        
    def fill_grid(self, data):
        """ fill the 1D waterdepths onto a 2D grid.  """
        if len(data) != len(self.template.rows):
            raise Exception("Template and waterdepth data do not match")
        grid = np.empty(shape=(len(self.template.y_array),len(self.template.x_array)))
        grid[:] = np.NaN
        for i in range(len(data)):
            grid[self.template.rows[i], self.template.cols[i]] = data[i]
        return grid

def dict_list_to_matrix(dict_list):
    """ convert list dictionaries to a matrix"""
    matrix = []
    name_list = []
    for key in dict_list.keys():
        matrix.append(dict_list[key])
        name_list.append(key)
    return [matrix, name_list]
        
    

def create_prob_map(data_class, treshold = 0.05):
    """ create a probablistic inundation map, by showing the probablities of exceeding an inundation treshold"""
    wd = data_class.max_waterdepths
    prob_map = np.zeros_like(wd[list(wd.keys())[0]])
    for key in wd.keys():
        inundated = np.greater(wd[key], treshold).astype(int)
        prob_map = np.add(prob_map, inundated)
    prob_map = np.divide(prob_map, len(wd))
    grid = data_class.fill_grid(prob_map)
    grid = np.ma.masked_where(grid == 0, grid)
    fig, ax = plt.subplots(figsize=(18, 18))
    plot = ax.pcolormesh(data_class.template.x_array, data_class.template.y_array, grid, cmap = 'plasma_r', vmin = 0, vmax = 1, shading = 'auto')
    cx.add_basemap(ax, crs= 'EPSG:28992', source = cx.providers.Esri.WorldImagery)

    ax.set_xlim([min(data_class.template.x_array) - (max(data_class.template.x_array)-min(data_class.template.x_array))*0.05 ,
                 max(data_class.template.x_array) + (max(data_class.template.x_array)-min(data_class.template.x_array))*0.05])
    ax.set_ylim([min(data_class.template.y_array) - (max(data_class.template.y_array)-min(data_class.template.y_array))*0.02 ,
                 max(data_class.template.y_array) + (max(data_class.template.y_array)-min(data_class.template.y_array))*0.02])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.05)
    cbar = plt.colorbar(plot, cax=cax, orientation = 'horizontal')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar.ax.tick_params(labelsize=23)
    cax.xaxis.set_ticks_position("top")
    cbar.set_label('Probablity of inundation larger than {} cm'.format(treshold*100), fontsize = 28, labelpad = -130)
    return fig

def create_percentile_map(data_class, percentile = 0.9): 
    """plot map with the waterdepths according to the xth percentile"""
    wd = data_class.max_waterdepths
    [matrix, name_list] = dict_list_to_matrix(wd)
    matrix = np.multiply(matrix, 100) # convert to cm
    result = np.percentile(matrix, q = percentile*100, axis = 0, interpolation = 'higher')
    grid = data_class.fill_grid(result)
    grid = np.ma.masked_where(grid == 0, grid)
    
    fig, ax = plt.subplots(figsize=(18, 18))
    plot = ax.pcolormesh(data_class.template.x_array, data_class.template.y_array, grid, cmap = 'rainbow', vmin = 0, vmax = 50, shading = 'auto')
    cx.add_basemap(ax, crs= 'EPSG:28992', source = cx.providers.Esri.WorldImagery)
    
    ax.set_xlim([min(data_class.template.x_array) - (max(data_class.template.x_array)-min(data_class.template.x_array))*0.05 ,
                 max(data_class.template.x_array) + (max(data_class.template.x_array)-min(data_class.template.x_array))*0.05])
    ax.set_ylim([min(data_class.template.y_array) - (max(data_class.template.y_array)-min(data_class.template.y_array))*0.02 ,
                 max(data_class.template.y_array) + (max(data_class.template.y_array)-min(data_class.template.y_array))*0.02])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.05)
    cbar = plt.colorbar(plot, cax=cax, orientation = 'horizontal')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar.ax.tick_params(labelsize=23)
    cax.xaxis.set_ticks_position("top")
    cbar.set_label('Inundation depth according to ' + str(int(percentile*100)) + "$^{th}$ percentile $[cm]$", fontsize = 28, labelpad = -130)
    
    return fig

def create_inundated_area(data_class, treshold = 0.05):
    """ Plot the inundated area and inundated volume per scenario"""
    wd = data_class.max_waterdepths
    inundated_volume = []
    inundated_area = []
    for key in wd.keys():
        inundated = np.greater(wd[key], treshold).astype(int)
        inundated_area.append(inundated.sum() * data_class.template.cell_size *  data_class.template.cell_size / 1000 / 1000) # in km^2
        volume = np.sum(wd[key]) * data_class.template.cell_size
        inundated_volume.append(volume)
    x_ticks = range(1, len(wd.keys()) + 1)
    x_labels = ["event " + str(x) for x in x_ticks]
    fig, ax1 = plt.subplots(figsize=(20, 7))
    area = ax1.plot(x_ticks, inundated_area, lw = 2, c = "#005F73", label = "Inundated area")
    ax2 = ax1.twinx()
    volume = ax2.plot(x_ticks, inundated_volume, lw = 2.5, c = "#AE2012", label = "Inundated volume", ls = ":")
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, rotation = 45, ha = 'right', rotation_mode='anchor')
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_xlim([min(x_ticks), max(x_ticks)])
    ax1.set_ylim([0, max(inundated_area) * 1.05])
    ax1.set_ylabel("Inundated area $[km^2]$", fontsize = 18)
    ax1.set_ylim([0, max(inundated_area)*1.1])
    ax2.set_ylim([0, max(inundated_volume)*1.2])
    ax2.set_ylabel("Inundated volume $[m^3]$", fontsize = 18, rotation = 270, labelpad = 30)
    ax1.legend(loc = 'upper left', fontsize = 16)
    ax2.legend(loc = 'upper right', fontsize = 16)
    return fig
        
        
def create_images(data_class, save_folder, treshold = 0.05, percentile = 0.9):
    """ create all images with one function"""
    prob_img = create_prob_map(data_class, treshold)
    prob_img.savefig(os.path.join(save_folder, "probibalistic.png"))
    
    area_img = create_inundated_area(data_class, treshold)
    area_img.savefig(os.path.join(save_folder, "area.png"))
    
    percentile_img = create_percentile_map(data_class, percentile = percentile)
    percentile_img.savefig(os.path.join(save_folder, "percentile.png"))


data_folder = r"/data/output"
save_folder = r"/data/images"

if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

data_class = model_data(data_folder)          
create_images(data_class, save_folder, treshold = 0.02, percentile = 0.9)

    

        
        
        
        