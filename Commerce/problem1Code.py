
import time
import pandas as pd
from lib import datasetPrimAnalysis, namestr, splitTimeSeriesData, createKey

def question1(df_li):
    '''
    Aggregate the Sales_Qty for each Store-SKU at a month level; detect any Outliers in the Sales_Qty for 
    each Store-SKU combination and apply an outlier treatment on the same. Specify the outlier treatment 
    technique.
    '''
    key = 'Secondary'
    
    t0 = int(time.time())
    print('Execution start at', t0)
    print('Analyzing and PreProcessing the Data')
    ## transfering the feature "SKU_Code" to object and then viewing the result
#     for df_name in df_li:
    print('"{}" dataframe shape:  {}'.format(key, df_li[key].shape))

    ## Changing feature data type to object
    df_li[key][['SKU_Code']] = df_li[key][['SKU_Code']].astype(str)

    ## Changing feature data type to datetime
    df_li[key]['Date'] = pd.to_datetime(df_li[key]['Date'],format='%Y-%m-%d') 

    ## sorting df based on Date
    df_li[key].sort_values(by=['Date'], inplace=True)
    df_li[key].reset_index(drop=True, inplace=True)

    # display(df_li[df_name].head())
    _ = datasetPrimAnalysis(df_li[key])
    print('****'*25,'\n\n')


    ## Loading Dataset to generate estimate on Quantity
    temp_sDF = df_li[key].copy()

    ## Working in the mentioned date range which lies between 1Jan2016 and 31 Dec 2017
    t1 = int(time.time())
    print('TimeTaken {} sec\n'.format(t1-t0))

    print('Selecting the Data according to the mentioned Date Range ("2016-01-01" - "2017-12-31")')
    temp_sDF = temp_sDF.loc[(temp_sDF['Date'] >= '2016-01-01') & (temp_sDF['Date'] <= '2017-12-31'), :]
    temp_sDF.reset_index(drop=True, inplace=True)

    temp_sDF = pd.DataFrame(temp_sDF.groupby(by= ['Store_Code', 'SKU_Code', 'Date'])['Sales_Qty', 'MRP', 'SP'].sum().sort_index()).reset_index()
    ## Get On basis of Monthly Level
    print('Processing Data Month-wise')
    temp_sDF['YrMonName'] =  temp_sDF['Date'].dt.strftime("%Y-%m")#%b
    tot = pd.DataFrame(temp_sDF.groupby(by= ['Store_Code', 'SKU_Code', 'YrMonName'])['Sales_Qty'].sum()).reset_index()

    # display(tot.head())

    ## Generate padding/ adding empty months
    t2 = int(time.time())
    print('TimeTaken {} sec\n'.format(t2-t1))
    print('Adding Observations for the month for which are data is not present in the dataset.')
    ## Adding Extra Empty Rows 
    tot.index = createKey(tot, ['Store_Code', 'SKU_Code', 'YrMonName'] )
    li = [ int(ele.split('-')[1]) for ele in tot['YrMonName'] ]
    elemToTrav = pd.Series([ ele.split('-')[0] for ele in tot.index ]).unique()
    ri_new = [ ele +'-{0:0>2}'.format(i) for ele in elemToTrav for i in range( min(li), max(li)+1) ]
    tot = tot.reindex(ri_new)
    tot['Store_Code'] = [ ele.split('|')[0] for ele in tot.index ]
    tot['SKU_Code'] = [ ele.split('|')[1] for ele in tot.index ]
    tot['YrMonName'] = [ ele.split('|')[2] for ele in tot.index ]
    tot.reset_index(drop=True, inplace=True)

    ## Assigning Value
    print('Filling values in the newly created observation.')
    tot['Sales_Qty'] = tot.groupby(by= ['Store_Code', 'SKU_Code'])['Sales_Qty'].fillna(0).astype('int')
    # display(tot.head(12))

    ## Detecting possible Outiler Case
    t3 = int(time.time())
    print('TimeTaken {} sec\n'.format(t3-t2))

    print('Checking for the outlier cases. \nNote: Since we have very less number of observations in Store-SKU pair and additionally we are not ablle to commment properly on the trend. Therefore finding outlier is a challenge')
    ## IQR won't work well because of the skewdness ppresent in the dataset
    ## Zscore is used

    tot['isOutlier'] = False
    for st in tot['Store_Code'].unique():
        for prod in tot.loc[tot['Store_Code']==st, 'SKU_Code'].unique():
            qt = tot.loc[(tot['Store_Code']==st) & (tot['SKU_Code']==prod), 'Sales_Qty']
            std, mean = qt.std(), qt.mean(),
            if std == 0: ## NoChange
                outlier = [ False for i in range(len(qt)) ]
            else:
                zscore = [ (ele - mean) / std for ele in qt ]
                ## Zscore based outlier detection is using Threshold 
                ### of 3 not good as Dataset is too small 
                outlier = list((pd.Series(zscore) > 4) |(pd.Series(zscore) < -4))  
            tot.loc[(tot['Store_Code']==st) & (tot['SKU_Code']==prod), 'isOutlier'] = outlier

    # display(tot.head(15))

    ## Filling Outlier Cases with mean
    t4 = int(time.time())
    print('TimeTaken {} sec\n'.format(t4-t3))

    print('Treating Outlier Observations.')
    outDF = tot.loc[tot['isOutlier'] == True, :]#.shape 
    for st in outDF['Store_Code'].unique():
        for prod in outDF['SKU_Code'].unique():
            val = tot.loc[(tot['Store_Code']==st) & (tot['SKU_Code']==prod), 'Sales_Qty' ].mean()
            tot.loc[(tot['Store_Code']==st) & (tot['SKU_Code']==prod) & (tot['isOutlier']), 'Sales_Qty' ] = val

    t5 = int(time.time())
    print('TimeTaken {} sec\n'.format(t5-t4))
    print('Whole Execution Time {} sec\n'.format(t5-t0))
    # display(tot.head(15))

    return tot


if __name__ == "__main__":
    secDf = pd.read_csv('data/WC_DS_Ex1_Sec_Sales.csv')
    df_li = {'Secondary': secDf }
    question1(df_li)
