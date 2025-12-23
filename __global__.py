
import platform
import os
from os.path import join

computer_name = platform.node()
centimeter_factor = 1 / 2.54
if 'wheat' in computer_name:
    # Wheat
    this_root = '/data/home/wenzhang/Yang/Alpha-Earth_Carbon_Flux/'
    global_device = 'cuda'
    print('Platform: Wheat')
elif 'yangli-ubt' in computer_name:
    # Dell
    this_root = '/home/yangli/SSD4T/Alpha-Earth_Carbon_Flux/'
    global_device = 'cuda'
    print('Platform: Dell')
elif 'Yang-M4Pro.local' in computer_name:
    # MacBook
    import matplotlib
    print('Platform: MacOS')
    global_device = 'mps'
    # this_root = '/Volumes/HDD/GPP_ML/'
    # this_root = '/Volumes/NVME4T/Prithvi_AGB/'
    this_root = '/Volumes/SSD4T/Alpha-Earth_Carbon_Flux/'
    matplotlib.use('TkAgg')
# elif 'yangligeo' in computer_name:
#     # yangligeo VPS
#     print('Platform: yangligeo')
#     this_root = '/root/GPP_ML/'
else:
    print('computer_name:',computer_name)
    raise ValueError('computer_name not recognized')
if not os.path.isdir(this_root):
    raise ValueError(f'working directory not found: {this_root}')

print('this_root:', this_root)
data_root = join(this_root, 'data')
results_root = join(this_root, 'results')
temp_root = join(this_root, 'temp')
conf_root = join(this_root, 'conf')


# global_res_gedi = 1000.89502334966744
# global_res_hls = 30
# global_nodata_value = -999999
# global_band_list_11 = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2", "DEM", "NDVI", "MNDWI", "NBR", "NDWI"]
# global_band_list_8 = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2", "DEM", "NDVI"]

def global_wkt_84():
    wkt_str = '''GEOGCRS["WGS 84",
ENSEMBLE["World Geodetic System 1984 ensemble",
    MEMBER["World Geodetic System 1984 (Transit)"],
    MEMBER["World Geodetic System 1984 (G730)"],
    MEMBER["World Geodetic System 1984 (G873)"],
    MEMBER["World Geodetic System 1984 (G1150)"],
    MEMBER["World Geodetic System 1984 (G1674)"],
    MEMBER["World Geodetic System 1984 (G1762)"],
    MEMBER["World Geodetic System 1984 (G2139)"],
    ELLIPSOID["WGS 84",6378137,298.257223563,
        LENGTHUNIT["metre",1]],
    ENSEMBLEACCURACY[2.0]],
PRIMEM["Greenwich",0,
    ANGLEUNIT["degree",0.0174532925199433]],
CS[ellipsoidal,2],
    AXIS["geodetic latitude (Lat)",north,
        ORDER[1],
        ANGLEUNIT["degree",0.0174532925199433]],
    AXIS["geodetic longitude (Lon)",east,
        ORDER[2],
        ANGLEUNIT["degree",0.0174532925199433]],
USAGE[
    SCOPE["Horizontal component of 3D system."],
    AREA["World."],
    BBOX[-90,-180,90,180]],
ID["EPSG",4326]]'''
    return wkt_str

