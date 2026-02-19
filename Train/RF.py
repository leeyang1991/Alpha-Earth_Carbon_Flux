import matplotlib.pyplot as plt
import numpy as np
from __init__ import *

this_script_root = join(results_root,'RF')

class Random_forests:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Random_forests', this_script_root, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe', 'dataframe.df')
        pass

    def run(self):
        # self.copy_monthly_df()
        # self.train_monthly()
        self.train_annual()
        pass

    def train_annual(self):
        from preprocess import GEE_Embedding
        outdir = join(self.this_class_arr,'train_annual_models')
        T.mkdir(outdir)
        # self.copy_df()
        df = self.__gen_df_init()
        x_variables = GEE_Embedding.Download_from_GEE().bands_list()
        y_variable = 'GPP_NT_VUT_REF'
        X = df[x_variables]
        Y = df[y_variable]
        clf, mse, r_model,r_model_train, score, score_train, Y_test, y_pred, Y_train, y_pred_train = self._random_forest_train(X, Y)
        model_path = join(outdir, f'{y_variable}_rf_model.pkl')
        T.save_dict_to_binary(clf, model_path)
        pass

    def train_monthly(self):
        from preprocess import GEE_Embedding
        outdir = join(self.this_class_arr,'train_monthly_models')
        T.mkdir(outdir)
        # self.copy_df()
        df_dir = join(self.this_class_arr, 'dataframe_monthly')
        for f in T.listdir(df_dir):
            dff = join(df_dir, f)
            if not dff.endswith('.df'):
                continue
            df = T.load_df(dff)
            x_variables = GEE_Embedding.Download_from_GEE().bands_list()
            y_variable = 'GPP_NT_VUT_REF'
            X = df[x_variables]
            Y = df[y_variable]
            clf, mse, r_model, r_model_train, score, score_train, Y_test, y_pred, Y_train, y_pred_train = self._random_forest_train(
                X, Y)
            model_name = f.replace('.df','_rf_model.pkl')
            model_path = join(outdir, model_name)
            T.save_dict_to_binary(clf, model_path)

    def copy_df(self):
        from preprocess import dataframe
        T.mkdir(join(self.this_class_arr, 'dataframe'))
        if isfile(self.dff):
            print('already exists: ', self.dff)
            print('press enter to overwrite')
            pause()
            pause()
            pause()
        T.mkdir(join(self.this_class_arr, 'dataframe'))
        dff = join(dataframe.Gen_Dataframe().this_class_arr, 'dataframe.df')
        df = T.load_df(dff)
        T.save_df(df,self.dff)
        T.df_to_excel(df, self.dff)

    def copy_monthly_df(self):
        from preprocess import dataframe
        outdir = join(self.this_class_arr, 'dataframe_monthly')
        T.mkdir(outdir)
        fdir = join(dataframe.Gen_Dataframe_monthly().this_class_arr,'monthly_df')
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            out_path = join(outdir,f)
            shutil.copy(fpath, out_path)

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff

    def _random_forest_train(self, X, Y):
        '''
        :param X: a dataframe of x variables
        :param Y: a dataframe of y variable
        :param variable_list: a list of x variables
        :return: details of the random forest model and the importance of each variable
        '''
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # split the data into training and testing
        clf = RandomForestRegressor(n_estimators=100, n_jobs=-1) # build a random forest model
        clf.fit(X_train, Y_train) # train the model
        y_pred = clf.predict(X_test) # predict the y variable using the testing data
        y_pred_train = clf.predict(X_train) # predict the y variable using the training data
        r_model = stats.pearsonr(Y_test, y_pred)[0] # calculate the correlation between the predicted y variable and the actual y variable
        r_model_train = stats.pearsonr(Y_train, y_pred_train)[0]
        mse = sklearn.metrics.mean_squared_error(Y_test, y_pred) # calculate the mean squared error
        score = clf.score(X_test, Y_test) # calculate the R^2
        score_train = clf.score(X_train, Y_train)
        # return clf, importances_dic, mse, r_model, score, Y_test, y_pred
        return clf, mse, r_model,r_model_train, score, score_train, Y_test, y_pred, Y_train, y_pred_train



    def __train_model(self,X,y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, random_state=1, test_size=0.0) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=20) # build a random forest model
        rf.fit(X, y) # train the model
        # r2 = rf.score(X_test,y_test)
        return rf,0.999


class Predict:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Predict', this_script_root, mode=2)
        pass

    def run(self):
        # self.predict_annual()
        # self.predict_monthly()
        # self.concat_predict_annual()
        self.concat_predict_monthly()
        pass

    def predict_annual(self):
        from preprocess import GEE_Embedding
        outdir = join(self.this_class_tif,'predict_annual')
        T.mkdir(outdir)
        model_path = join(Random_forests().this_class_arr,'train_annual_models','GPP_NT_VUT_REF_rf_model.pkl')
        clf = T.load_dict_from_binary(model_path)
        tif_dir = join(GEE_Embedding.Download_from_GEE().this_class_arr,'mosaic')
        for year in T.listdir(tif_dir):
            outdir_i = join(outdir, year)
            T.mkdir(outdir_i, force=True)
            site_flag = 0
            total_site_flag = len(T.listdir(join(tif_dir,year)))
            for site in T.listdir(join(tif_dir,year)):
                outf = join(outdir_i, f'{site}.tif')
                site_flag += 1
                if isfile(outf):
                    continue
                data_list = []
                profile = ''
                for f in T.listdir(join(tif_dir,year,site)):
                    fpath = join(tif_dir,year,site,f)
                    data,profile = RasterIO_Func().read_tif(fpath)
                    data_list.append(data)
                data_list = np.array(data_list)
                predicted_array = np.ones_like(data_list) * np.nan
                predicted_array = predicted_array[0]
                n_rows, n_cols = data_list.shape[1], data_list.shape[2]

                params_list = []
                for i in range(n_rows):
                    params = (n_cols, data_list, clf, i)
                    params_list.append(params)


                results = MULTIPROCESS(self.kernel_predict, params_list).run(process=24,desc=f'{site_flag}/{total_site_flag} in {year}')
                for i in range(len(results)):
                    arr_i, ii = results[i]
                    predicted_array[ii] = arr_i

                RasterIO_Func().write_tif(predicted_array, outf, profile)

    def predict_monthly(self):
        from preprocess import GEE_Embedding
        outdir = join(self.this_class_tif, 'predict_monthly')
        T.mkdir(outdir)

        tif_dir = join(GEE_Embedding.Download_from_GEE().this_class_arr, 'mosaic')

        month_list = list(range(1,13))
        year_list = T.listdir(tif_dir)
        site_list = T.listdir(join(tif_dir,year_list[0]))

        model_dict = {}
        for mon in month_list:
            model_path = join(Random_forests().this_class_arr, 'train_monthly_models',
                              f'GPP_NT_VUT_REF_{mon:02d}_rf_model.pkl')
            clf = T.load_dict_from_binary(model_path)
            model_dict[mon] = clf

        params_list = []

        for site in site_list:
            outdir_i = join(outdir, site)
            for year in year_list:
                for mon in month_list:
                    params = [model_dict, outdir_i, site, tif_dir, year, mon]
                    params_list.append(params)
                    # self.kernel_predict_monthly(params)

        MULTIPROCESS(self.kernel_predict_monthly, params_list).run(process=24)

    def kernel_predict_monthly(self, params):
        model_dict,outdir_i,site,tif_dir,year,mon = params
        clf = model_dict[mon]
        try:
            T.mkdir(outdir_i, force=True)
        except:
            pass
        outf = join(outdir_i, f'{year}-{mon:02d}.tif')
        if isfile(outf):
            return
        data_list = []
        profile = ''
        for f in T.listdir(join(tif_dir, year, site)):
            fpath = join(tif_dir, year, site, f)
            data, profile = RasterIO_Func().read_tif(fpath)
            data_list.append(data)
        data_list = np.array(data_list)
        predicted_array = np.ones_like(data_list) * np.nan
        predicted_array = predicted_array[0]
        n_rows, n_cols = data_list.shape[1], data_list.shape[2]

        for i in range(n_rows):
            for j in range(n_cols):
                x_values = data_list[:, i, j]
                if np.any(np.isinf(x_values)):
                    continue
                x_values = x_values.reshape(1, -1)
                y_pred = clf.predict(x_values)[0]
                predicted_array[i, j] = y_pred

        RasterIO_Func().write_tif(predicted_array, outf, profile)

    def concat_predict_annual(self):
        fdir = join(self.this_class_tif, 'predict_annual')
        outdir = join(self.this_class_tif, 'concat_predict_annual')
        T.mkdir(outdir)
        site_list = []
        for year in T.listdir(fdir):
            for site in T.listdir(join(fdir,year)):
                if not site.endswith('.tif'):
                    continue
                site_list.append(site)
            break
        fail_num = 0
        for site in tqdm(site_list):
            # if not site == 'CA-ARB.tif':
            #     continue
            array_3d = []
            profile = ''
            for year in T.listdir(join(fdir)):
                fpath = join(fdir,year,site)
                data,profile = RasterIO_Func().read_tif(fpath)
                # print(data.shape)
                array_3d.append(data)
            outf = join(outdir,site)
            try:
                array_3d = np.array(array_3d)
                bands_description = T.listdir(join(fdir))
                RasterIO_Func().write_tif_multi_bands(array_3d, outf, profile, bands_description)
            except:
                print(site)
                fail_num += 1
        print(fail_num)

        pass

    def concat_predict_monthly(self):
        fdir = join(self.this_class_tif, 'predict_monthly')
        outdir = join(self.this_class_tif, 'concat_predict_monthly')
        T.mkdir(outdir)

        fail_num = 0
        for site in tqdm(T.listdir(fdir)):
            date_list = []
            for date in T.listdir(join(fdir,site)):
                if not date.endswith('.tif'):
                    continue
                date_str = date.replace('.tif','')
                date_list.append(date_str)

            array_3d = []
            profile = ''
            for date in tqdm(date_list):
                fpath = join(fdir,site,date+'.tif')
                data,profile = RasterIO_Func().read_tif(fpath)
                # print(data.shape)
                array_3d.append(data)
            outf = join(outdir,site + '.tif')
            try:
                array_3d = np.array(array_3d)
                bands_description = date_list
                RasterIO_Func().write_tif_multi_bands(array_3d, outf, profile, bands_description)
            except:
                print(site)
                fail_num += 1
        print(fail_num)

        pass

    def kernel_predict(self,params):
        n_cols, data_list, clf, i = params
        y_pred_list = []
        for j in range(n_cols):
            x_values = data_list[:, i, j]
            if np.any(np.isinf(x_values)):
                y_pred_list.append(np.nan)
                continue
            x_values = x_values.reshape(1, -1)
            y_pred = clf.predict(x_values)[0]
            # print(y_pred)
            y_pred_list.append(y_pred)
        y_pred_list = np.array(y_pred_list)
        return y_pred_list,i



def main():
    # Random_forests().run()
    Predict().run()
    pass

if __name__ == '__main__':
    main()