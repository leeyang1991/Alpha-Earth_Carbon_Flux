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
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Benchmark', this_script_root, mode=2)
        pass

    def run(self):
        # self.obs_vs_predict_monthly()
        # self.corr_obs_pred()
        # self.pred_obs_scatter_plot()
        # self.corr_obs_pred_statistic()
        self.plot_spatial_map()
        pass

    def obs_vs_predict_monthly(self):
        flux_site_dff = join(data_root,'Flux/dataframe/GPP_NT_VUT_REF_MM.df')
        pred_dir = join(results_root,'RF/Predict/tif/concat_predict_monthly')
        outdir = join(self.this_class_png,'obs_vs_predict_monthly')
        out_dir_result = join(self.this_class_arr,'obs_vs_predict_monthly')
        T.mkdir(out_dir_result,force=True)
        T.mkdir(outdir,force=True)
        flux_site_df = T.load_df(flux_site_dff)
        site_list = flux_site_df['SITE_ID'].tolist()
        print(outdir)

        for site in tqdm(site_list):
            df_site = flux_site_df[flux_site_df['SITE_ID']==site]
            gpp_fpath = join(pred_dir, site + '.tif')
            if not isfile(gpp_fpath):
                continue
            data, profile = RasterIO_Func().read_tif(gpp_fpath)
            band_list = RasterIO_Func_Extend().get_tif_description(gpp_fpath)
            date_obj_list_pred = []
            for band in band_list:
                year, mon = band.split('-')[0], band.split('-')[1]
                date_obj = datetime.datetime(int(year), int(mon), 1)
                date_obj_list_pred.append(date_obj)
            gpp_pred_val_list = []
            for i in range(data.shape[0]):
                data_i = data[i]
                date_i = date_obj_list_pred[i]
                mean_val = np.nanmean(data_i)
                gpp_pred_val_list.append(mean_val)

            col_name_list = df_site.columns.tolist()
            date_obj_list_obs = []
            obs_gpp_val_list = []
            for col_name in col_name_list:
                if col_name == 'SITE_ID':
                    continue
                col_name = str(col_name)
                year = col_name[:4]
                mon = col_name[4:6]
                date_obj = datetime.datetime(int(year), int(mon), 1)
                gpp = df_site[int(col_name)].values[0]
                if np.isnan(gpp):
                    continue
                obs_gpp_val_list.append(gpp)
                date_obj_list_obs.append(date_obj)
            all_date_obj_list = date_obj_list_obs + date_obj_list_pred
            all_date_obj_list = sorted(list(set(all_date_obj_list)))

            obs_dict = T.dict_zip(date_obj_list_obs,obs_gpp_val_list)
            pred_dict = T.dict_zip(date_obj_list_pred,gpp_pred_val_list)

            result_obs_list = []
            result_pred_list = []
            for date_obj in all_date_obj_list:
                if not date_obj in obs_dict:
                    obs = np.nan
                else:
                    obs = obs_dict[date_obj]
                if not date_obj in pred_dict:
                    pred = np.nan
                else:
                    pred = pred_dict[date_obj]
                result_obs_list.append(obs)
                result_pred_list.append(pred)
            df_result = pd.DataFrame({'date':all_date_obj_list,'obs_gpp':result_obs_list,'pred_gpp':result_pred_list})
            out_dff = join(out_dir_result,f'{site}.df')
            T.save_df(df_result,out_dff)
            T.df_to_excel(df_result,out_dff)

            plt.plot(date_obj_list_obs, obs_gpp_val_list,'-o', label='Observed GPP', color='b')
            plt.plot(date_obj_list_pred, gpp_pred_val_list,'-o', label='Predicted GPP', color='r')
            plt.xlabel('Date')
            plt.ylabel('GPP')
            plt.title(site)
            plt.legend()
            # plt.show()
            # exit()
            outf = join(outdir, f'{site}.pdf')
            plt.savefig(outf)

            outf = join(outdir, f'{site}.png')
            plt.savefig(outf)
            plt.close()
        pass

    def get_flux_site_metadata(self):
        flux_site_metadata_dff = join(data_root, 'Flux/metadata/metadata.df')
        flux_site_metadata_df = T.load_df(flux_site_metadata_dff)
        # T.print_head_n(flux_site_metadata_df)
        metadata_dict = T.df_to_dic(flux_site_metadata_df, key_str='SITE_ID')

        return metadata_dict,flux_site_metadata_df

    def corr_obs_pred(self):
        fdir = join(self.this_class_arr,'obs_vs_predict_monthly')
        outdir = join(self.this_class_png,'corr_obs_pred')
        T.mkdir(outdir,force=True)
        metadata_dict,_ = self.get_flux_site_metadata()
        key_temp = list(metadata_dict.keys())[0]
        pprint(metadata_dict[key_temp])
        # exit()
        point_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.df'):
                continue
            # print(f)
            site_name = f.split('.')[0]
            fpath = join(fdir,f)
            df = T.load_df(fpath)
            df = df.dropna(how='any')
            if len(df)<10:
                continue
            obs_gpp = df['obs_gpp'].values
            pred_gpp = df['pred_gpp'].values
            r,p = T.nan_correlation(obs_gpp,pred_gpp)
            lon = metadata_dict[site_name]['LOCATION_LONG']
            lat = metadata_dict[site_name]['LOCATION_LAT']
            dict_i = {'r':float(r),'p':float(p),'SITE_ID':site_name}
            point_list.append([lon,lat,dict_i])
        outSHPfn = join(outdir,'corr_obs_pred.shp')
        T.point_to_shp(point_list, outSHPfn)
        pass


    def get_df_unique_val_list(self, df, var_name):
        var_list = df[var_name]
        var_list = var_list.dropna()
        var_list = list(set(var_list))
        var_list.sort()
        var_list = tuple(var_list)
        return var_list


    def is_unique_key_in_df(self, df, unique_key):
        len_df = len(df)
        unique_key_list = self.get_df_unique_val_list(df, unique_key)
        len_unique_key = len(unique_key_list)
        if len_df == len_unique_key:
            return True
        else:
            return False

    def add_dic_to_df(self, df, dic, unique_key):
        # todo: add to lytools, bug fixed
        if not self.is_unique_key_in_df(df, unique_key):
            raise UserWarning(f'{unique_key} is not a unique key')
        all_val_list = []
        all_col_list = []
        for i, row in df.iterrows():
            unique_key_ = row[unique_key]
            if not unique_key_ in dic:
                dic_i = {}
                for col in dic[list(dic.keys())[0]]:
                    dic_i[col] = np.nan
            else:
                dic_i = dic[unique_key_]
            col_list = []
            val_list = []
            for col in dic_i:
                val = dic_i[col]
                col_list.append(col)
                val_list.append(val)
            all_val_list.append(val_list)
            all_col_list.append(col_list)
            # val_list.append(val)
        all_val_list = np.array(all_val_list)
        all_col_list = np.array(all_col_list)
        all_val_list_T = all_val_list.T
        all_col_list_T = all_col_list.T
        for i in range(len(all_col_list_T)):
            df[all_col_list_T[i][0]] = all_val_list_T[i]
        return df

    def pred_obs_scatter_plot(self):
        fdir = join(self.this_class_arr, 'obs_vs_predict_monthly')
        outdir = join(self.this_class_png, 'pred_obs_scatter_plot')
        T.mkdir(outdir, force=True)
        obs_gpp_list = []
        pred_gpp_list = []
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.df'):
                continue
            # print(f)
            site_name = f.split('.')[0]
            fpath = join(fdir, f)
            df = T.load_df(fpath)
            df = df.dropna(how='any')
            if len(df) < 10:
                continue
            obs_gpp = df['obs_gpp'].values
            pred_gpp = df['pred_gpp'].values
            for i in range(len(obs_gpp)):
                obs_gpp_list.append(obs_gpp[i])
                pred_gpp_list.append(pred_gpp[i])
        plt.figure(figsize=(4, 4))
        plt.scatter(obs_gpp_list, pred_gpp_list, alpha=0.3,color='gray')
        plt.xlabel('Observed GPP')
        plt.ylabel('Predicted GPP')
        # plt.title('Scatter Plot of Observed vs Predicted GPP')
        # 1:1 line
        min_val = min(min(obs_gpp_list), min(pred_gpp_list))
        max_val = max(max(obs_gpp_list), max(pred_gpp_list))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        # outf = join(outdir, 'obs_vs_pred_scatter.pdf')
        # plt.axis('equal')
        plt.ylim([0, 15])
        plt.xlim([0, 15])
        r2 = np.corrcoef(obs_gpp_list, pred_gpp_list)[0, 1] ** 2
        plt.text(0.05, 0.95, f'$R^2$ = {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        outf = join(outdir, 'obs_vs_pred_scatter.png')
        plt.savefig(outf,dpi=1300)
        print(outdir)

        plt.show()

    def corr_obs_pred_statistic(self):
        fdir = join(self.this_class_arr,'obs_vs_predict_monthly')
        outdir = join(self.this_class_png,'corr_obs_pred_statistic')
        T.mkdir(outdir,force=True)
        metadata_dict, metadata_df = self.get_flux_site_metadata()
        key_temp = list(metadata_dict.keys())[0]
        metadata_df = metadata_df.dropna(subset=['SITE_ID'])
        # pprint(metadata_dict[key_temp])
        # exit()
        result_dict = {}
        for f in T.listdir(fdir):
            if not f.endswith('.df'):
                continue
            # print(f)
            site_name = f.split('.')[0]
            fpath = join(fdir,f)
            df = T.load_df(fpath)
            df = df.dropna(how='any')
            if len(df)<10:
                continue
            obs_gpp = df['obs_gpp'].values
            pred_gpp = df['pred_gpp'].values
            r,p = T.nan_correlation(obs_gpp,pred_gpp)
            r2 = r * r
            dict_i = {'r2':float(r2),'p':float(p)}
            result_dict[site_name] = dict_i
        result_df = self.add_dic_to_df(metadata_df, result_dict, 'SITE_ID')
        result_df = result_df.dropna(subset=['r2'])
        T.print_head_n(result_df)
        r_list = result_df['r2'].values
        plt.hist(r_list,bins=20)
        plt.xlabel('Correlation Coefficient (r2)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Correlation Coefficients')
        outf1 = join(outdir, 'r2_distribution.pdf')
        plt.savefig(outf1)
        plt.close()

        IGBP_df_dict = T.df_groupby(result_df, 'IGBP')
        IGBP_dict = {}
        for IGBP in IGBP_df_dict:
            df_IGBP = IGBP_df_dict[IGBP]
            r2_value = df_IGBP['r2'].values
            r2_value_mean = np.nanmean(r2_value)
            IGBP_dict[IGBP] = r2_value_mean
        IGBP_dict = T.sort_dict_by_value(IGBP_dict)
        IGBP_list = list(IGBP_dict.keys())
        r2_value_mean_list = list(IGBP_dict.values())
        plt.bar(IGBP_list,r2_value_mean_list)
        plt.xlabel('IGBP')
        plt.ylabel('Mean Correlation Coefficient (r2)')
        plt.title('Mean Correlation Coefficient by IGBP')
        plt.xticks(rotation=45)
        # plt.show()
        outf2 = join(outdir, 'r2_by_IGBP.pdf')
        plt.savefig(outf2)
        plt.close()

        MAT_list = result_df['MAT'].values
        MAP_list = result_df['MAP'].values
        r2_list = result_df['r2'].values
        plt.scatter(MAT_list,MAP_list,c=r2_list,cmap='RdBu_r',s=50)
        plt.colorbar(label='Correlation Coefficient (r2)')
        plt.xlabel('Mean Annual Temperature (MAT)')
        plt.ylabel('Mean Annual Precipitation (MAP)')
        plt.title('Correlation Coefficient by MAT and MAP')
        # plt.show()
        outf3 = join(outdir, 'r2_by_MAT_MAP.pdf')
        plt.savefig(outf3)
        plt.close()
        print(outdir)
        pass

    def plot_spatial_map(self):

        site = 'US-SRG'
        outdir = join(self.this_class_png, 'spatial_map',site)
        # exit()
        T.mkdir(outdir, force=True)
        fdir = join(results_root,'RF/Predict/tif/predict_monthly',site)
        for f in T.listdir(fdir):
            if not f.startswith('2022'):
                continue
            date = f.split('.')[0]
            fpath = join(fdir,f)
            data, profile = RasterIO_Func().read_tif(fpath)
            data_mean = np.nanmean(data,axis=0)
            plt.imshow(data,cmap='Greens',vmin=0,vmax=7)
            # plt.axis('off')
            plt.colorbar()
            plt.yticks([],[])
            plt.xticks([],[])
            plt.title(f'{site} {date}')
            # plt.show()
            # exit()
            # print(fpath)
            outf = join(outdir, f'{date}.pdf')
            plt.savefig(outf)
            plt.close()
            exit()
        print(outdir)

        pass

def main():
    # PDSI_process().run()
    # Drought_response().run()
    Benchmark().run()


if __name__ == '__main__':
    main()

