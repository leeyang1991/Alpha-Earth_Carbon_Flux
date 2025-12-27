from __init__ import *
result_root_this_script = join(results_root,'dataframe')


class Gen_Dataframe:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Gen_Dataframe', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe.df')
        pass

    def run(self):
        # self.load_average_embedding()


        df = self.__load_df()
        df = self.add_carbon_flux(df)
        #
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)
        pass

    def load_average_embedding(self):
        from preprocess import GEE_Embedding
        if isfile(self.dff):
            print('dataframe exists!')
            pause()
            print('dataframe exists!')
            pause()
            print('dataframe exists!')
            pause()
        embedding_dff = join(GEE_Embedding.Download_from_GEE().this_class_arr,'average_embedding/average_embedding.df')
        df_embedding = T.load_df(embedding_dff)
        T.print_head_n(df_embedding)
        print('len(df_embedding):',len(df_embedding))
        T.save_df(df_embedding, self.dff)
        T.df_to_excel(df_embedding, self.dff)

    def add_carbon_flux(self,df):
        import Flux
        carbon_dff = join(Flux.Fluxdata().data_dir,'dataframe/GPP_NT_VUT_REF_YY.df')
        carbon_df = T.load_df(carbon_dff)
        carbon_dict = T.df_to_dic(carbon_df,key_str='SITE_ID')
        # pprint(carbon_dict)
        # exit()
        # T.print_head_n(carbon_df)
        flux_val_list = []
        for i,row in tqdm(df.iterrows(),desc='adding carbon flux',total=len(df)):
            site = row['site']
            year = row['year']
            year = int(year)
            if not year in carbon_dict[site]:
                flux_val_list.append(np.nan)
                continue
            flux_val = carbon_dict[site][year]
            flux_val_list.append(flux_val)

        df['GPP_NT_VUT_REF'] = flux_val_list
        df = df.dropna(subset=['GPP_NT_VUT_REF'])
        T.print_head_n(df)

        return df

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
        return df

class Gen_Dataframe_monthly:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Gen_Dataframe_monthly', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe.df')
        pass

    def run(self):
        # self.load_average_embedding()


        df = self.__load_df()
        df = self.add_monthly_carbon_flux(df)
        # #
        # T.save_df(df, self.dff)
        # T.df_to_excel(df, self.dff)
        pass

    def load_average_embedding(self):
        from preprocess import GEE_Embedding
        if isfile(self.dff):
            print('dataframe exists!')
            pause()
            print('dataframe exists!')
            pause()
            print('dataframe exists!')
            pause()
        embedding_dff = join(GEE_Embedding.Download_from_GEE().this_class_arr,'average_embedding/average_embedding.df')
        df_embedding = T.load_df(embedding_dff)
        T.print_head_n(df_embedding)
        print('len(df_embedding):',len(df_embedding))
        T.save_df(df_embedding, self.dff)
        T.df_to_excel(df_embedding, self.dff)

    def add_monthly_carbon_flux(self,df):
        outdir = join(self.this_class_arr,'monthly_df')
        T.mk_dir(outdir)
        import Flux
        carbon_dff = join(Flux.Fluxdata().data_dir,'dataframe/GPP_NT_VUT_REF_MM.df')
        carbon_df = T.load_df(carbon_dff)
        carbon_dict = T.df_to_dic(carbon_df,key_str='SITE_ID')
        # pprint(carbon_dict['US-xTL'])
        # exit()
        # T.print_head_n(carbon_df)
        for m in range(1,13):
            month = f'{m:02d}'
            flux_val_list = []
            for i,row in tqdm(df.iterrows(),desc=month,total=len(df)):
                site = row['site']
                year = row['year']
                year = int(year)
                col_name = f'{year}{m:02d}'
                col_name = int(col_name)
                if not col_name in carbon_dict[site]:
                    flux_val_list.append(np.nan)
                    continue
                flux_val = carbon_dict[site][col_name]
                flux_val_list.append(flux_val)
            df['GPP_NT_VUT_REF'] = flux_val_list
            df = df.dropna(subset=['GPP_NT_VUT_REF'])
            outf = join(outdir,f'GPP_NT_VUT_REF_{month}.df')
            T.save_df(df, outf)
            T.df_to_excel(df, outf)

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
        return df


def main():
    # Gen_Dataframe().run()
    Gen_Dataframe_monthly().run()
    pass


if __name__ == '__main__':
    main()
    pass