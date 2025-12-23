from __init__ import *
result_root_this_script = join(results_root,'dataframe')


class Gen_Dataframe:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Gen_Dataframe', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe.df')
        pass

    def run(self):
        df = self.average_embedding()
        # df = self.__load_df()
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)
        pass

    def average_embedding(self):
        if isfile(self.dff):
            print('dataframe exists!')
            pause()
            print('dataframe exists!')
            pause()
            print('dataframe exists!')
            pause()
        embedding_fdir = join()

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



def main():
    Gen_Dataframe().run()
    pass


if __name__ == '__main__':

    pass