# coding=utf-8
import shutil

import numpy as np
import urllib3
from __init__ import *
import ee
import math
import geopandas as gpd
from geopy import Point
from geopy.distance import distance as Distance
from shapely.geometry import Polygon

this_script_root = join(data_root,'Embedding')
# this_script_root = '/Volumes/NVME4T/GPP_ML/data/HLS/'
# exit()

class Expand_points_to_rectangle:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Expand_points_to_rectangle',
            this_script_root, mode=2)
        pass

    def run(self):
        point_list,name_list = self.read_point_shp()
        rectangle_list = self.expand_points_to_rectangle(point_list)
        self.write_rectangle_shp(rectangle_list,name_list)
        pass

    def read_point_shp(self):

        flux_dff = join(data_root,'Flux/dataframe/GPP_NT_VUT_REF_MM.df')
        flux_df = T.load_df(flux_dff)
        site_list = flux_df['SITE_ID'].tolist()
        flux_metadata_dff = join(data_root,'Flux/metadata/metadata.df')
        flux_metadata_df = T.load_df(flux_metadata_dff)
        T.print_head_n(flux_metadata_df)
        flux_metadata_dic = T.df_to_dic(flux_metadata_df,'SITE_ID')
        lon_list = [flux_metadata_dic[site_id]['LOCATION_LONG'] for site_id in site_list]
        lat_list = [flux_metadata_dic[site_id]['LOCATION_LAT'] for site_id in site_list]
        point_list = zip(lon_list, lat_list)
        point_list = list(point_list)
        name_list = site_list
        return point_list,name_list

    def expand_points_to_rectangle(self, point_list):
        distance_i = 25*30/1000. # km
        # print(point_list)
        rectangle_list = []
        for point in point_list:
            lon = point[0]
            lat = point[1]
            p = Point(latitude=lat, longitude=lon)
            north = Distance(kilometers=distance_i).destination(p, 0)
            south = Distance(kilometers=distance_i).destination(p, 180)
            east = Distance(kilometers=distance_i).destination(p, 90)
            west = Distance(kilometers=distance_i).destination(p, 270)
            # rectangle = Polygon([(west.longitude, west.latitude), (east.longitude, east.latitude),
            #                         (north.longitude, north.latitude), (south.longitude, south.latitude)])
            # east = (east.longitude, east.latitude)
            # west = (west.longitude, west.latitude)
            # north = (north.longitude, north.latitude)
            # south = (south.longitude, south.latitude)

            east_lon = east.longitude
            west_lon = west.longitude
            north_lat = north.latitude
            south_lat = south.latitude

            ll_point = (west_lon, south_lat)
            lr_point = (east_lon, south_lat)
            ur_point = (east_lon, north_lat)
            ul_point = (west_lon, north_lat)

            polygon_geom = Polygon([ll_point, lr_point, ur_point, ul_point])

            rectangle_list.append(polygon_geom)
        return rectangle_list

    def write_rectangle_shp(self, rectangle_list,name_list):
        outdir = join(self.this_class_arr, 'sites')
        T.mkdir(outdir)
        outf = join(outdir, 'sites.shp')
        crs = {'init': 'epsg:4326'}  # 设置坐标系
        polygon = gpd.GeoDataFrame(crs=crs, geometry=rectangle_list)  # 将多边形对象转换为GeoDataFrame对象
        polygon['name'] = name_list

        # 保存为shp文件
        polygon.to_file(outf)
        pass

    def GetDistance(self,lng1, lat1, lng2, lat2):
        radLat1 = self.rad(lat1)
        radLat2 = self.rad(lat2)
        a = radLat1 - radLat2
        b = self.rad(lng1) - self.rad(lng2)
        s = 2 * math.asin(math.sqrt(
            math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
        s = s * 6378.137 * 1000
        distance = round(s, 4)
        return distance

        pass

    def rad(self,d):
        return d * math.pi / 180

class Download_from_GEE:

    def __init__(self):
        '''
        band: A00, ... , A63
        https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
        '''
        self.collection = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL' # derived from Sentinel
        # --------------------------------------------------------------------------------
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            f'Download_from_GEE',
            this_script_root, mode=2)

        # ee.Authenticate()
        ee.Initialize(project='lyfq-263413')

        # pause()
        # exit()

    def run(self):
        # year_list = list(range(2018,2025))
        # self.download_images(2024)
        # for year in year_list:
        #     self.download_images(year)
        # self.check()
        # self.unzip()
        self.merge_bands()
        pass


    def download_images(self,year=1982):
        outdir = join(self.this_class_arr,'GEE_download',str(year))
        T.mk_dir(outdir,force=True)
        startDate = f'{year}-01-01'
        endDate = f'{year+1}-01-01'

        rectangle_f = join(Expand_points_to_rectangle().this_class_arr, 'sites/sites.shp')
        rectangle_df = gpd.read_file(rectangle_f)
        geometry_list = rectangle_df['geometry'].tolist()
        site_list = rectangle_df['name'].tolist()

        param_list = []
        for i,geo in enumerate(geometry_list):
            param = (site_list,i,outdir,geo,startDate,endDate)
            param_list.append(param)
            # self.kernel_download_from_gee(param)
        MULTIPROCESS(self.kernel_download_from_gee,param_list).run(process=20,process_or_thread='t',desc=f'download_{year}')

    def kernel_download_from_gee(self,param):
        site_list,i,outdir,geo,startDate,endDate = param
        site = site_list[i]
        # print(site)
        outdir_i = join(outdir, site)
        T.mk_dir(outdir_i)
        ll = geo.bounds[0:2]
        ur = geo.bounds[2:4]
        region = ee.Geometry.Rectangle(ll[0], ll[1], ur[0], ur[1])

        Collection = ee.ImageCollection(self.collection)
        Collection = Collection.filterDate(startDate, endDate).filterBounds(region)

        info_dict = Collection.getInfo()
        # pprint.pprint(info_dict)
        # print(len(info_dict['features']))
        # exit()
        ids = info_dict['features']
        for i in ids:
            dict_i = eval(str(i))
            # pprint.pprint(dict_i['id'])
            # exit()
            outf_name = dict_i['id'].split('/')[-1] + '.zip'
            out_path = join(outdir_i, outf_name)
            if isfile(out_path):
                continue
            # print(outf_name)
            # exit()
            # print(dict_i['id'])
            # l8 = l8.median()
            # l8_qa = l8.select(['QA_PIXEL'])
            # l8_i = ee.Image(dict_i['LANDSAT/LC08/C02/T1_L2/LC08_145037_20200712'])
            Image = ee.Image(dict_i['id'])
            # Image_product = Image.select('total_precipitation')
            bands_list = self.bands_list()
            Image_product = Image.select(bands_list)
            # print(Image_product);exit()
            # region = [-111, 32.2, -110, 32.6]# left, bottom, right,
            # region = [-180, -90, 180, 90]  # left, bottom, right,
            exportOptions = {
                'scale': 10,
                'maxPixels': 1e13,
                'region': region,
                # 'fileNamePrefix': 'exampleExport',
                # 'description': 'imageToAssetExample',
            }
            url = Image_product.getDownloadURL(exportOptions)

            try:
                self.download_i(url, out_path)
            except:
                print('download error', out_path)
                continue
        pass



    def download_i(self,url,outf):
        # try:
        http = urllib3.PoolManager()
        r = http.request('GET', url, preload_content=False)
        body = r.read()
        with open(outf, 'wb') as f:
            f.write(body)


    def bands_list(self):
        band_name_list = []
        for i in range(64):
            band_name = f'A{i:02d}'
            band_name_list.append(band_name)
        return band_name_list

    def unzip(self):
        fdir = join(self.this_class_arr,'GEE_download')
        outdir = join(self.this_class_arr,'unzip')
        T.mk_dir(outdir,force=True)
        for folder in T.listdir(fdir):
            print(folder)
            fdir_i = join(fdir,folder)
            outdir_i = join(outdir,folder)
            T.mkdir(outdir_i)
            for site in T.listdir(join(fdir_i)):
                fdir_ii = join(fdir_i,site)
                outdir_ii = join(outdir_i,site)
                T.mkdir(outdir_ii)
                T.unzip(fdir_ii,outdir_ii)
            # exit()
        pass


    def check(self):
        fdir = join(self.this_class_arr,'GEE_download')
        for year in T.listdir(fdir):
            for site in tqdm(T.listdir(join(fdir,year)),desc=year):
                for f in T.listdir(join(fdir,year,site)):
                    fpath = join(fdir,year,site,f)
                    try:
                        zipfile.ZipFile(fpath, 'r')
                    except:
                        os.remove(fpath)
                        print(fpath)
                        continue
                    pass
        pass

    def mosaic(self):

        pass

    def raster2array(self, rasterfn):
        '''
        create array from raster
        Agrs:
            rasterfn: tiff file path
        Returns:
            array: tiff data, an 2D array
        '''
        raster = gdal.Open(rasterfn)
        projection_wkt = raster.GetProjection()
        geotransform = raster.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
        array = np.asarray(array)
        del raster
        return array, originX, originY, pixelWidth, pixelHeight,projection_wkt

    def array2raster(self, newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array, projection_wkt,ndv=-999999):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = longitude_start
        originY = latitude_start
        # open geotiff
        driver = gdal.GetDriverByName('GTiff')
        if os.path.exists(newRasterfn):
            os.remove(newRasterfn)
        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
        # outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_UInt16)
        # ndv = 255
        # Add Color Table
        # outRaster.GetRasterBand(1).SetRasterColorTable(ct)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        # Write Date to geotiff
        outband = outRaster.GetRasterBand(1)

        outband.SetNoDataValue(ndv)
        outband.WriteArray(array)
        outRasterSRS = osr.SpatialReference()
        # outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(projection_wkt)
        # Close Geotiff
        outband.FlushCache()
        del outRaster

    def gdal_merge_bands(self,tif_list,bands_name_list,outf):
        src0 = gdal.Open(tif_list[0])
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(outf,
                               src0.RasterXSize,
                               src0.RasterYSize,
                               len(tif_list),
                               gdal.GDT_Float32)

        out_ds.SetGeoTransform(src0.GetGeoTransform())
        out_ds.SetProjection(src0.GetProjection())
        for idx, tif in enumerate(tif_list, start=1):
            src = gdal.Open(tif)
            band = src.GetRasterBand(1).ReadAsArray()
            out_ds.GetRasterBand(idx).WriteArray(band)
            out_ds.GetRasterBand(idx).SetDescription(bands_name_list[idx - 1])

        out_ds.FlushCache()
        out_ds = None


def main():
    # Expand_points_to_rectangle().run()
    Download_from_GEE().run()

    pass

if __name__ == '__main__':
    main()