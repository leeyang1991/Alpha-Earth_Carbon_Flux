import matplotlib.pyplot as plt

from utils import *
T = Tools()
this_script_root = join(results_root,'Statistic')

class PDSI_process:

    def __init__(self):
        self.datadir = join(data_root,'Terraclimate/PDSI')

        pass

    def run(self):
        self.clip_by_sites()
        pass

    def clip_by_sites(self):
        fdir = join(self.datadir,'tif')
        sites_dir = join(results_root,'RF/Predict/tif/concat_predict_annual')
        outdir = join(self.datadir,'tif_clip')
        T.mkdir(outdir,force=True)

        for site in tqdm(T.listdir(sites_dir)):
            if not site.endswith('.tif'):
                continue
            fpath = join(sites_dir,site)
            bounds = RasterIO_Func_Extend().get_tif_bounds_(fpath)
            outdir_i = join(outdir,site.split('.')[0])
            T.mkdir(outdir_i,force=True)
            for f in T.listdir(fdir):
                fpath = join(fdir,f)
                outf = join(outdir_i,f)
                RasterIO_Func_Extend().clip_tif_by_bounds(fpath,outf,bounds)

        pass


class Drought_response:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Drought_response', this_script_root, mode=2)
        pass

    def run(self):
        # self.GPP_monthly_anomaly()
        # self.plot_time_series()
        # self.plot_spatial_corr()
        pass

    def GPP_monthly_anomaly(self):
        fdir = join(results_root,'RF/Predict/tif/concat_predict_monthly')
        outdir = join(self.this_class_tif,'GPP_monthly_anomaly')
        T.mkdir(outdir,force=True)
        params_list = []
        for f in T.listdir(fdir):
            params = [fdir,f,outdir]
            # self.kernel_GPP_monthly_anomaly(params)
            params_list.append(params)
        MULTIPROCESS(self.kernel_GPP_monthly_anomaly,params_list).run(process=24)


    def get_tif_description(self,fpath):
        with rasterio.open(fpath) as src:
            band_list = src.descriptions
        return band_list

    def kernel_GPP_monthly_anomaly(self,params):
        fdir,f,outdir = params
        fpath = join(fdir, f)
        with rasterio.open(fpath) as src:
            band_list = src.descriptions
        data3d, profile = RasterIO_Func().read_tif(join(fdir, f))
        rows, cols = data3d.shape[1], data3d.shape[2]
        data3d_anomaly = np.full_like(data3d, np.nan)
        for r in range(rows):
            for c in range(cols):
                data_pixel = data3d[:, r, c]
                if np.isnan(data_pixel).all():
                    continue
                pix_anomaly = Pre_Process().z_score_climatology(data_pixel)
                data3d_anomaly[:, r, c] = pix_anomaly
        outf = join(outdir, f)
        RasterIO_Func().write_tif_multi_bands(data3d_anomaly, outf, profile, band_list)

        pass

    def plot_time_series(self):
        gpp_anomaly_dir = join(self.this_class_tif,'GPP_monthly_anomaly')
        outdir = join(self.this_class_png,'GPP_PDSI_time_series')
        T.mkdir(outdir,force=True)
        pdsi_dir = join(data_root,'Terraclimate/PDSI/tif_clip')
        for site in tqdm(T.listdir(pdsi_dir)):
            gpp_anomaly_f = join(gpp_anomaly_dir, site + '.tif')
            band_list = self.get_tif_description(gpp_anomaly_f)

            date_obj_list = []
            for band in band_list:
                year,mon = band.split('-')[0], band.split('-')[1]
                date_obj = datetime.datetime(int(year),int(mon),1)
                date_obj_list.append(date_obj)

            gpp_anomaly3d, _ = RasterIO_Func().read_tif(gpp_anomaly_f)
            pdsi_3d = []
            for f in T.listdir(join(pdsi_dir,site)):
                fpath = join(pdsi_dir,site,f)
                pdsi_data, _ = RasterIO_Func().read_tif(fpath)
                pdsi_3d.append(pdsi_data)
            pdsi_3d = np.stack(pdsi_3d)

            gpp_anomaly_mean_list = []
            pdsi_mean_list = []
            for i in range(gpp_anomaly3d.shape[0]):
                gpp_anomaly = gpp_anomaly3d[i]
                pdsi = pdsi_3d[i]
                gpp_anomaly_mean = np.nanmean(gpp_anomaly)
                pdsi_mean = np.nanmean(pdsi)
                gpp_anomaly_mean_list.append(gpp_anomaly_mean)
                pdsi_mean_list.append(pdsi_mean)
            plt.plot(date_obj_list,gpp_anomaly_mean_list,label='GPP Anomaly',color='g')
            plt.ylim(-2,2)
            plt.twinx()
            plt.plot(date_obj_list,pdsi_mean_list,label='PDSI',color='r')
            plt.ylim(-10,10)
            plt.hlines(0,date_obj_list[0],date_obj_list[-1],colors='k',linestyles='dashed')
            plt.xlabel('Date')
            plt.ylabel('Mean Anomaly')
            plt.title(site)
            plt.legend()
            # plt.show()
            outf = join(outdir, f'{site}.png')
            plt.savefig(outf)
            plt.close()

        pass


    def plot_spatial_corr(self):
        gpp_anomaly_dir = join(self.this_class_tif, 'GPP_monthly_anomaly')
        outdir = join(self.this_class_png, 'plot_spatial_corr')
        T.mkdir(outdir, force=True)
        pdsi_dir = join(data_root, 'Terraclimate/PDSI/tif_clip')
        # corr_list = []
        # x_list = []
        # y_list = []
        point_list = []
        for site in tqdm(T.listdir(pdsi_dir)):
            gpp_anomaly_f = join(gpp_anomaly_dir, site + '.tif')
            band_list = self.get_tif_description(gpp_anomaly_f)

            date_obj_list = []
            for band in band_list:
                year, mon = band.split('-')[0], band.split('-')[1]
                date_obj = datetime.datetime(int(year), int(mon), 1)
                date_obj_list.append(date_obj)
            bounds = RasterIO_Func_Extend().get_tif_bounds_(gpp_anomaly_f)
            center_lon = (bounds[0] + bounds[2]) / 2
            center_lat = (bounds[1] + bounds[3]) / 2

            gpp_anomaly3d, _ = RasterIO_Func().read_tif(gpp_anomaly_f)
            pdsi_3d = []
            for f in T.listdir(join(pdsi_dir, site)):
                fpath = join(pdsi_dir, site, f)
                pdsi_data, _ = RasterIO_Func().read_tif(fpath)
                pdsi_3d.append(pdsi_data)
            pdsi_3d = np.stack(pdsi_3d)

            gpp_anomaly_mean_list = []
            pdsi_mean_list = []
            for i in range(gpp_anomaly3d.shape[0]):
                gpp_anomaly = gpp_anomaly3d[i]
                pdsi = pdsi_3d[i]
                gpp_anomaly_mean = np.nanmean(gpp_anomaly)
                pdsi_mean = np.nanmean(pdsi)
                gpp_anomaly_mean_list.append(gpp_anomaly_mean)
                pdsi_mean_list.append(pdsi_mean)
            r,p = T.nan_correlation(gpp_anomaly_mean_list,pdsi_mean_list)
            dict_i = {'r':float(r),'p':float(p)}
            point_list.append([center_lon,center_lat,dict_i])
        outSHPfn = join(outdir,'spatial_corr.shp')
        T.point_to_shp(point_list, outSHPfn)
        # exit()
        pass


class Benchmark:

    def __init__(self):

        pass

def main():
    # PDSI_process().run()
    Drought_response().run()


if __name__ == '__main__':
    main()

