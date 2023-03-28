from rasterstats import zonal_stats
import os
import pandas as pd
import numpy as np
import rasterio as rasterio
import geopandas as gpd
from datetime import datetime
from datetime import timedelta
import shutil
import pickle as pkl
import requests
import json
import zipfile


#%% Collect data
class get_ensemble_forecast:
    def __init__(self, start_time, duration, folder, accumulated = False, interval = 1, ensemble_members = ["1", "2"]):
        self.model_date = start_time.strftime("%Y%m%d%H%M%S")
        self.start_date = (start_time + timedelta(hours = 1)).strftime("%Y%m%d%H%M%S")
        self.end_date = (start_time + duration).strftime("%Y%m%d%H%M%S")
        self.accumulated = accumulated
        self.interval = 1
        self.ensemble_members = ensemble_members
        
        self.folder = folder
        self.data_folder = os.path.join(self.folder, 'prec')
        if os.path.exists(os.path.join(self.folder, 'prec')) == False:
            os.mkdir(os.path.join(self.folder, 'prec'))
        self.zip_file_loc = os.path.join(self.data_folder, 'reponse.zip')
        shutil.rmtree(self.data_folder) # delete all previous data
        self.prepare_folder()
        self.response = self.get_ensemble_members()
        self.write_response_to_zip()
        self.unzip_response()
        self.place_and_rename_files()
     
    def prepare_folder(self):
        if os.path.exists(self.data_folder) == False:
            os.mkdir(self.data_folder)
        
    def get_ensemble_members(self):
        print("Fetching data...")
        url = "https://hnapi.hydronet.com/api/grids/get"
        payload = {
            "Readers": [
                {
                    "DataSourceCode": "Knmi.Harmonie.Mos.Ensemble",
                    "Settings": {
                        "StructureType": "EnsembleGrid",
                        "EnsembleNames": self.ensemble_members,
                        "IgnorePyramids": True,
                        "ModelDate": self.model_date,
                        "ReadQuality": False,
                        "ReadAvailability": False,
                        "StartDate": self.start_date,
                        "EndDate": self.end_date,
                        "VariableCodes": ["P"],
                        "Interval": {
                            "Type": "Hours",
                            "Value": self.interval
                        },
                        "ReadAccumulated": self.accumulated,
                        "Extent": {
                            "Xll": 124434.875,
                            "Yll": 459485.71875,
                            "Xur": 129495.484375,
                            "Yur": 464845.65625,
                            "SpatialReference": {"Epsg": 28992}
                        }
                    }
                }
            ],
            "Exporter": {
                "DataFormatCode": "geotiff.wkt",
                "Settings": {"Formatting": "Indented"},
                "SpatialReference": {"Epsg": 4329}
            }
        }
        headers = {"Authorization": "$HYDRONET AUTHORIZATION$", "Content-Type": "application/json"
        }
        response = requests.request("POST", url, json=payload, headers=headers)
        return response

    def write_response_to_zip(self):
        with open(self.zip_file_loc, "wb") as f:
            f.write(self.response.content)
        return 0

    def unzip_response(self):
        print("Unzipping...")
        with zipfile.ZipFile(self.zip_file_loc, 'r') as zip_ref:
            zip_ref.extractall(self.data_folder)
        os.remove(self.zip_file_loc)
        return 0
    
    def place_and_rename_files(self):
        print("Placing and renaming...")
        files = os.listdir(self.data_folder)
        for file in files:
            contents = file.split("_")
            if len(contents) == 6 and file.startswith("Knmi.Harmonie.Mos.Ensemble_precipitation_"):
                member = int(contents[3])
                date = datetime.strptime(contents[4], "%Y-%m-%dT%Hh%Mm%Ss")
                new_name = "NSL_" + date.strftime("%Y%m%d%H%M") + ".tif"
                if os.path.exists(os.path.join(self.data_folder, str(member))) == False:
                    os.mkdir(os.path.join(self.data_folder, str(member)))
                os.rename(os.path.join(self.data_folder, file), os.path.join(self.data_folder, str(member), new_name))
            else:
                raise Exception("File naming is different than expected in scripting! \n{}".format(name))
        return 0
            
class get_evaporation_forecast:
    """ Er is nog geen voorspellings data. Daarom gebruik ik voor nu de gemiddelde per maand de afgelopen 10 jaar"""
    def __init__(self, start_time, duration, folder):
        self.start_time = start_time
        self.duration = duration
        self.folder = os.path.join(folder, 'evap')
        if os.path.exists( self.folder) == False:
            os.mkdir( self.folder)
        self.evaporation = self.get_evaporation()
        
    
    def get_evaporation(self):
        evo = {1: 0.267910447761194,
                2: 0.570572207084468,
                3: 1.2487562189054722,
                4: 2.2071979434447297,
                5: 2.8580645161290317,
                6: 3.300769230769231,
                7: 3.3330769230769226,
                8: 2.7080645161290318,
                9: 1.8455555555555543,
                10: 0.9537634408602151,
                11: 0.4124999999999999,
                12: 0.22177419354838712} # there are averages per month over past 10 years
        output = {}
        for i in range(self.duration.days + 2):
            time = self.start_time.date() + timedelta(days = i)
            output[time] = evo[int(time.month)]
            
        with open(os.path.join(self.folder, "evaporation_data.pkl"), 'wb') as f:
            pkl.dump(output, f)    
        return output
    

#%% functions from delft3dfmpy (een beetje aangepast)
def read_raster(file, static=False):
    """
    Method to read a raster. All rasterio types are accepted, plus IDF: in that case the iMod-package is used to read the IDF raster (IDF is cusomary for MODFLOW/SIMGRO models.)

    Parameters
    ----------
    file : raster
        
    static : BOOL, optional
        If static than no time information needs to be deduced.

    Returns
    -------
    rasterio grid and an affine object.        

    """
    if not static:         
        time = pd.Timestamp(os.path.split(file)[1].split('_')[1].split('.')[0])
    if str(file).lower().endswith('idf'):        
        dataset = imod.idf.open(file)
        header = imod.idf.header(file,pattern=None)        
        grid = dataset[0,0,:,:].values        
        affine =from_origin(header['xmin'], header['ymax'], header['dx'], header['dx'])                            
    else:        
        dataset = rasterio.open(file)
        affine = dataset.transform
        grid = dataset.read(1)
    if static:
        return grid, affine
    else:
        return grid, affine, time
    
def generate_precip(areas, precip_folder):
    """
    Method to obtain catchment-average seepage fluxes from rasters. The time step is deduced from the raster filenames.
    comes from delft3dfmpy
    """
    # warnings.filterwarnings('ignore')
    file_list = os.listdir(precip_folder)
    times = []        
    arr = np.zeros((len(file_list), len(areas.code)))
    for ifile, file in enumerate(file_list):
        array, affine, time = read_raster(os.path.join(precip_folder, file))
        times.append(time)            
        stats = zonal_stats(areas, array, affine=affine, stats="mean", all_touched=True)
        arr[ifile,:]= [s['mean'] for s in stats]        
    result = pd.DataFrame(arr, columns=['ms_'+str(area) for area in areas.code])
    result.index = times 
    return result

def generate_evap(areas, evap_folder):    
    """
    Method to obtain catchment-average evaporation fluxes from rasters. The time step is deduced from the raster filenames. Since only one timeeries is allowed, the meteo areas are dissolved to a user specifield field.
    comes from delft3dfmpy
    """
    # warnings.filterwarnings('ignore')
    file_list = os.listdir(evap_folder)
    # aggregated evap
    #areas['dissolve'] = 1
    #agg_areas = areas.iloc[0:len(areas),:].dissolve(by='dissolve',aggfunc='mean')
    times = []    
    arr = np.zeros((len(file_list), len(areas)))
    for ifile, file in enumerate(file_list):
        array, affine, time = read_raster(os.path.join(evap_folder, file))
        times.append(time)                  
        stats = zonal_stats(areas, array, affine=affine, stats="mean",all_touched=True)
        arr[ifile,:] = [s['mean'] for s in stats]       
    result = pd.DataFrame(arr, columns=['ms_'+str(area) for area in areas.code])
    result.index = times 
    return result

#%% modify precipitaion
class update_precipitation:
    """ Read the new precipitaion data, and modify the model file accordingly """
    def __init__(self, start_time, duration, folder_precip, folder_rr, folder_meteo_areas, rain_multiplier = 1, add_rain = 0):
        self.start_time = start_time
        self.duration = duration
        self.folder_rr = folder_rr
        self.folder_precip_database = folder_precip
        self.folder_meteo_areas = folder_meteo_areas
        self.working_dir = os.path.join(os.getcwd(), 'working-dir')
        self.rain_multiplier = rain_multiplier
        self.add_rain = add_rain
        
        if os.path.exists(self.working_dir) == False:
            os.mkdir(self.working_dir)
        else:
            shutil.rmtree(self.working_dir)
            os.mkdir(self.working_dir)
            
        self.select_correct_precip()
        
        self.precipitation = self.precip_from_raster()
        self.modify_precip_file()
        
        shutil.rmtree(self.working_dir)
        
    
    def select_correct_precip(self):
        """ From the precipitaion files, read the correct files that describe rianfall within the simulation period"""
        for file in os.listdir(self.folder_precip_database):
            name = file[4:-4]
            date = datetime.strptime(name, "%Y%m%d%H%M")
            if date >= self.start_time:
                if date < self.start_time + self.duration:
                    shutil.copyfile(os.path.join(self.folder_precip_database, file),
                                    os.path.join(self.working_dir, file))
        if len(os.listdir(self.working_dir)) != int(self.duration.total_seconds()/3600):
            raise Exception("Not all precipitation data could be included. Amount of files is {} while duration in hours is {}".format(len(os.listdir(self.working_dir)), int(self.duration.total_seconds()/3600)))
        return 0
            
    def precip_from_raster(self):
        """ Convert from raster to a list of precipitation per RR meteo station """
        meteo_areas = gpd.read_file(self.folder_meteo_areas)
        df_precip = generate_precip(meteo_areas, self.working_dir)
        result = df_precip.to_numpy()
        result = np.add(np.multiply(result, self.rain_multiplier), self.add_rain)
        result = np.round(result, 3)
        return result
    
    def modify_precip_file(self):
        """ Modify the precipitation file. Change de duration, and the precipitation content of the default file (default as in the file that is generated by initiating the model """
        real_name = "DEFAULT.BUI"
        if os.path.exists(os.path.join(self.folder_rr, real_name)) == False:
            raise Exception("DEFAULT.BUI is not the specified folder")
        
        with open(os.path.join(self.folder_rr, real_name), 'r+') as f:
            lines = f.readlines()
            for iline, line in enumerate(lines):
                if line.startswith('*Het format is:'):
                    start_index = iline
            heading = lines[:start_index + 1]
            start_date_model = lines[start_index + 1].split(' ')[:5]
            start_date_model = datetime(int(start_date_model[0]), int(start_date_model[1]), int(start_date_model[2]))
            start_time_model = start_date_model + timedelta(seconds = int(self.start_time.hour * 3600 + self.start_time.minute * 60 + self.start_time.second))
            start_time_model = start_time_model.strftime("%Y %m %d %H %M %S").replace(' 0', ' ')
            duration = str(self.duration.days) + " " + str(int((self.duration.total_seconds() -self.duration.days*3600*24)/3600)) + " 0 0" 
            time_line = start_time_model + " " + duration
        
        with open(os.path.join(self.folder_rr, real_name), 'w') as f:
            for line in heading:
                f.write(line)
            f.write(time_line)
            f.write('\n')
            for row in self.precipitation:
                line = ' '.join([str(e) for e in row])
                f.write(line)
                f.write('\n')
        return 0


#%% change evaporation file
class update_evaporation:
    """ Read the new evaporation data, and modify the model file accordingly """
    def __init__(self, start_time, duration, folder_evap, folder_rr):
        self.start_time = start_time
        self.duration = duration
        self.folder_rr = folder_rr
        self.folder_evap_database = folder_evap
        self.working_dir = os.path.join(os.getcwd(), 'working-dir')
        
        self.evaporation = self.read_evaporation()
        
        self.modify_evap_file()
        
    def read_evaporation(self):
        """ Read the new evaporation data"""
        with open(os.path.join(self.folder_evap_database, 'evaporation_data.pkl'), 'rb') as f:
            r = pkl.load(f)
        output = []
        for key in r.keys():
            if key >= self.start_time.date():
                if key <= (self.start_time + self.duration + timedelta(days = 1)).date():
                    output.append(str(round(r[key], 3)))
        daystart = self.start_time.day
        dayend = (self.start_time + self.duration).day
        if len(output) < dayend - daystart + 1:
            raise Exception("Loaded {} evaporation files which is to few since simulation spans from day {} to day {}".format(len(output), daystart, dayend))
        return output
    
    def modify_evap_file(self):
        """ Modfiy the evaporation file. Starting date remains same as original, but amount of days is changed to match simulation duraiton. Evaporation over all stations is modified """
        real_name = "DEFAULT.EVP"
        if os.path.exists(os.path.join(self.folder_rr, real_name)) == False:
            raise Exception("DEFAULT.EVP is not the specified folder")
        
        with open(os.path.join(self.folder_rr, real_name), 'r+') as f:
            lines = f.readlines()
            for iline, line in enumerate(lines):
                if line.startswith('*jaar maand dag verdamping'):
                    start_index = iline
            heading = lines[:start_index + 1]
            start_date = lines[start_index + 1].split()[:3]
            start_date = datetime(int(start_date[0]), int(start_date[1]), int(start_date[2]))
        
        with open(os.path.join(self.folder_rr, real_name), 'w') as f:
            for line in heading:
                f.write(line)
            for index, evap in enumerate(self.evaporation):
                date = start_date + timedelta(days = index)
                line = date.strftime("%Y    %m    %d").replace(' 0', ' ') + "   " + str(evap)
                f.write(line)
                f.write('\n')
        return 0


class modify_model:
    """Modify the requried components of the model.
    The goal is to be able to change the evaporation, precipiptation, and the duration over which the model operates
    The start date of hte model is kept the same (many things have to be changed otherwise), but the duration and start time (only time not date) can be changed with this function """
    def __init__(self, start_time, duration, folder_model, folder_precip, folder_evap, folder_meteo_areas, rain_multiplier = 1, add_rain = 0):
        self.start_time = start_time
        self.duration = duration
        self.model_folder = folder_model
        self.folder_meteo_areas = folder_meteo_areas
        self.fm_folder = os.path.join(self.model_folder, 'fm')
        self.rr_folder = os.path.join(self.model_folder, 'rr')
        self.folder_precip = folder_precip
        self.folder_evap = folder_evap
        self.rain_multiplier = rain_multiplier
        self.add_rain = add_rain
        
        self.modify_fm()
        self.modify_rr()
        
        self.update_precipitation = update_precipitation(self.start_time, self.duration, self.folder_precip, self.rr_folder, self.folder_meteo_areas, self.rain_multiplier, self.add_rain)
        self.update_evaporation = update_evaporation(self.start_time, self.duration, self.folder_evap, self.rr_folder)
        
    def modify_fm(self):
        """ modify the mdu file to have correct start and stop time """
        with open(os.path.join(self.fm_folder, 'demo.mdu'), 'r+') as f:
            lines = f.readlines()
            for iline, line in enumerate(lines):
                if line.startswith('RefDate '):
                    refdate_index = iline
                if line.startswith('TStart '):
                    tstart_index = iline
                if line.startswith('TStop '):
                    tstop_index = iline
        

        tstart = int(self.start_time.hour * 3600 + self.start_time.minute * 60 + self.start_time.second)
        lines[tstart_index] = "TStart                            = {}                   # Start time w.r.t. RefDate (in TUnit)\n".format(tstart)
        tstop = int(tstart + self.duration.total_seconds())
        lines[tstop_index] = "TStop                             = {}           # Stop  time w.r.t. RefDate (in TUnit)\n".format(tstop)
        
        with open(os.path.join(self.fm_folder, 'demo.mdu'), 'w') as f:
            for line in lines:
                f.write(line)
        return 0
    
    def modify_rr(self):
        """ modify the rr setup file to have correct start and stop time. Start time is original start date with the new start time. End date is start time + duration"""
        with open(os.path.join(self.rr_folder, 'DELFT_3B.INI'), 'r+') as f:
            lines = f.readlines()
            for iline, line in enumerate(lines):
                if line.startswith('StartTime'):
                    tstart_index = iline
                if line.startswith('EndTime'):
                    tstop_index = iline
        
        tstart_string = lines[tstart_index].split('=')[-1].strip('\n').strip("'")
        tstart = datetime.strptime(tstart_string, '%Y/%m/%d;%H:%M:%S').date()
        tstart = datetime(tstart.year, tstart.month, tstart.day)
        tstart = tstart + timedelta(seconds = int(self.start_time.hour * 3600 + self.start_time.minute * 60 + self.start_time.second))
        tstart_line = tstart.strftime("%Y/%m/%d;%H:%M:%S")
        lines[tstart_index] = "StartTime='{}'\n".format(tstart_line)
        tstop = (tstart + self.duration).strftime("%Y/%m/%d;%H:%M:%S")
        lines[tstop_index] = "EndTime='{}'\n".format(tstop)
        
        with open(os.path.join(self.rr_folder, 'DELFT_3B.INI'), 'w') as f:
            for line in lines:
                f.write(line)
        return 0
    
class model_setup_controller:
    def __init__(self, folder, start_time, duration, rain_multiplier = 1, add_rain = 0):
        self.start_time = start_time
        self.duration = duration
        self.main_folder = folder
        self.rain_multiplier = rain_multiplier
        self.add_rain = add_rain
        self.folder_meteo_areas = os.path.join(folder, 'meteo-areas')
        self.folder_data = os.path.join(folder, 'data-loader')
        self.folder_model_setup = os.path.join(folder, 'model-setup')
        self.folder_model_specifics = os.path.join(folder, 'model-specifics')
        self.folder_precip = os.path.join(self.folder_data, 'prec')
        self.folder_evap = os.path.join(self.folder_data, 'evap')
        if os.path.exists(self.folder_model_specifics) == False:
            os.mkdir(self.folder_model_specifics)

        self.ensembles = [int(x) for x in os.listdir(self.folder_precip)]
        self.create_model_specific_dirs()
        
        self.files_to_modify = self.get_files_to_modify()
        self.place_file_specific_dirs()
        self.modify_model_specific_files()
        
    
    def get_files_to_modify(self):
        folder_mdu = 'fm'
        path_mdu = os.path.join(self.folder_model_setup, folder_mdu)
        cnt = 0
        for file in os.listdir(path_mdu):
            if file.endswith('.mdu'):
                mdu_file = file
                cnt += 1
        if cnt > 1:
            raise Exception("Found multiple mdu files")
        mdu = [folder_mdu + "/" + mdu_file]
        rr_files = ['rr/DEFAULT.BUI', 'rr/DEFAULT.EVP', 'rr/DELFT_3B.INI']
        files = rr_files + mdu
        return files
    
    def create_model_specific_dirs(self):
        print("Creating directories for model specific files...")
        for ens in self.ensembles:
            folder = os.path.join(self.folder_model_specifics, str(ens))
            if os.path.exists(folder) == False:
                os.mkdir(folder)
        return 0

    def place_file_specific_dirs(self):
        print("Placing files in model specific directories...")
        from_dir = os.path.join(self.folder_model_setup)
        for ens in self.ensembles:
            goal_dir = os.path.join(self.folder_model_specifics, str(ens))
            for file in self.files_to_modify:
                src = os.path.join(from_dir, file)
                dest = os.path.join(goal_dir, file)
                model_setup_controller.copy_file_and_create_dir(src, dest)
        return 0

    def copy_file_and_create_dir(src, dest):
        loc = os.path.dirname(dest)
        if os.path.exists(loc) == False:
            os.mkdir(loc)
        shutil.copyfile(src, dest)
        return 0

    def modify_model_specific_files(self):
        for e in os.listdir(self.folder_model_specifics):
            folder = os.path.join(self.folder_model_specifics, e)
            print("Modifying files in {}".format(folder))
            modify_model(start_time = self.start_time,
                         duration = self.duration,
                         folder_model = folder,
                         folder_precip = os.path.join(self.folder_precip, e),
                         folder_evap = self.folder_evap,
                         folder_meteo_areas = self.folder_meteo_areas,
                         rain_multiplier = self.rain_multiplier,
                         add_rain = self.add_rain)
        return 0
            
        
    
#%%

def read_settings(folder):
    file = os.path.join(folder, 'settings.txt')
    keys = ['start_model_ensemble ', 'duration_ensemble_forecast_data', 'start_time_model', 'duration_model', 'main_folder', 'multiplication_rain', 'add_rain', 'ensembles_from', 'ensembles_to']
    with open(file, 'r') as f:
        lines = f.readlines()
    
    settings = {}
    for l in lines:
        if l[0] != '#':
            elements = l.split('=')
            for i in range(len(elements)):
                elements[i] = elements[i].strip().replace('\n', '')
            if elements[0] == 'start_model_ensemble':
                settings[elements[0]] = datetime.strptime(elements[1], '%Y/%m/%d_%H:%M:%S')
            if elements[0] == 'duration_ensemble_forecast_data':
                settings[elements[0]] = timedelta(hours  = int(elements[1]))
            if elements[0] == 'start_time_model':
                settings[elements[0]] = datetime.strptime(elements[1], '%Y/%m/%d_%H:%M:%S')
            if elements[0] == 'duration_model':
                settings[elements[0]] = timedelta(hours  = int(elements[1]))   
            if elements[0] == 'main_folder':
                settings[elements[0]] = elements[1]
            if elements[0] == 'multiplication_rain':
                settings[elements[0]] = float(elements[1])
            if elements[0] == 'add_rain':
                settings[elements[0]] = float(elements[1])
            if elements[0] == 'ensembles_from':
                settings[elements[0]] = int(elements[1])
            if elements[0] == 'ensembles_to':
                settings[elements[0]] = int(elements[1])  
    return settings
            


######### General settings #########
load_new_data = True
folder = r"/app/fedde-prepare-model"
settings = read_settings(folder)
main_folder = settings['main_folder']

######### import ensemble forecast settings #########
model_ensemble_forecast_date = settings['start_model_ensemble']
duration_ensemble_forecast_data =  settings['duration_ensemble_forecast_data']
ensemble_members = range(settings['ensembles_from'], settings['ensembles_to'] + 1)
ensemble_members = [str(x) for x in ensemble_members]
######### Model settings #########
start_time = settings['start_time_model']
duration = settings['duration_model']
rain_multiplier = settings['multiplication_rain']
add_rain = settings['add_rain']
##################################
folder_data = os.path.join(main_folder, 'data-loader')
if os.path.exists(folder_data) == False:
    os.mkdir(folder_data)


if load_new_data:
    a = get_ensemble_forecast(start_time = model_ensemble_forecast_date,
                              duration = duration_ensemble_forecast_data, 
                              folder = folder_data,
                              ensemble_members =  ensemble_members)
    
    b = get_evaporation_forecast(start_time = model_ensemble_forecast_date,
                              duration = duration_ensemble_forecast_data , 
                              folder = folder_data)



a = model_setup_controller(folder = main_folder, start_time = start_time, duration = duration, rain_multiplier = rain_multiplier, add_rain = add_rain)

