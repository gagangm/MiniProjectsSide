import pandas as pd


## Doing Primary analysis
def getMissingPct(df):
    '''    '''
    return [ round((df['IsNullSum'][i] / max(df['count']))*100,2) for i in range(len(df)) ]

def genMsg(txt):
    '''    '''
    print('_'*12+'| Number of feature/s which are {} : {} |'.format(*txt)+'_'*12)

def datasetPrimAnalysis(DF):
    '''
    Function which is used to analyze features and provide data insight
    '''
    df_explore = DF.copy()
    print('Overall dataset shape :', df_explore.shape)
    
    ## Creating a dataset that explain feature/s
    temp = pd.DataFrame(df_explore.isnull().sum(), columns = ['IsNullSum'])
    temp['dtypes'] = df_explore.dtypes.tolist()
    temp['IsNaSum'] = df_explore.isna().sum().tolist()
    
    ## Analyzing Time based Features
    temp_tim = temp.loc[temp['dtypes']=='datetime64[ns]' ,:]
    if (len(temp_tim) > 0):
        df_tim = df_explore.loc[:,temp_tim.index].fillna('Missing-NA')
        genMsg(['Time based', df_tim.shape[1]])
        temp_tim = temp_tim.join(df_tim.describe().T).fillna('')
        temp_tim['%Missing'] = getMissingPct(temp_tim)
        display(temp_tim)
    
    
    ## Analyzing Qualitative Features
    temp_cat = temp.loc[temp['dtypes']=='O' ,:]
    if (len(temp_cat) > 0):
        df_cat = df_explore.loc[:,temp_cat.index].fillna('Missing-NA')
        genMsg(['Qualitative', df_cat.shape[1]])
        temp_cat = temp_cat.join(df_cat.describe().T).fillna('')
        temp_cat['CategoriesName'] = [ list(df_cat[fea].unique()) for fea in temp_cat.index ]
        temp_cat['%Missing'] = getMissingPct(temp_cat)
        display(temp_cat)
    
    
    ## Analyzing Quantitative Features
    temp_num = temp.loc[((temp['dtypes']=='int') | (temp['dtypes']=='float')),:]
    if (len(temp_num) > 0):
        df_num = df_explore.loc[:,temp_num.index]#.fillna('Missing-NA')
        genMsg(['Quantitative', df_num.shape[1]])
        temp_num = temp_num.join(df_num.describe().T).fillna('')
        temp_num['%Missing'] = getMissingPct(temp_num)
        display(temp_num)
    # if temp_cat['dtypes'][i] == 'float', 'int', 'O'

    if len(temp)!= len(temp_tim) + len(temp_cat) + len(temp_num):
        print("Some columns data is missing b/c of data type")
    
    dit = {'TimeBased': temp_tim, 'Caterorical': temp_cat, 'Numerical': temp_num}
    return dit


## splitting the dataset

def namestr(obj, namespace): #namestr(primDf, globals())[0]
    return [name for name in namespace if namespace[name] is obj]

def splitTimeSeriesData(df, split_date, msg =True):
    ''' Split DataFramee based on Split Date
    '''
    df_train = df.loc[df['Date'] <= split_date]
    df_test = df.loc[df['Date'] > split_date]
    if msg: print('Original DataFrame Shape: {} \n\tTrain Shape: {}\tTest Shape:{}'.format(
        df.shape, df_train.shape, df_test.shape))
    return df_train, df_test



## Key Generator
def createKey(DF, Key_ColToUse):
    '''
    Use to combine columns to generate a key which is seperated by '|'
    '''
    df = DF.copy()
    for col_ind in range(len(Key_ColToUse)):
        I1 = df.index.tolist()
        I2 = df[Key_ColToUse[col_ind]].astype('str').tolist()
        if col_ind == 0:
            df.index = I2
        else:
            df.index = [ "|".join([I1[ind], I2[ind]]) for ind in range(len(I1)) ] #, I3[ind]
    return df.index 

