import time
import numpy as np
import pandas as pd
from lib import datasetPrimAnalysis, namestr, splitTimeSeriesData, createKey

def question4(df_li):
    '''
    Estimate the level of promotions (Discount%) for each Category-Store level at a month level - remove 
    any outliers / inconsistencies from this, and specify the technique used; the level of promotions is 
    defined as   Discount%=(1−sumofSP/sumofMRP)
    '''
    key = [ 'Secondary', 'Primary', 'Inventory' ]
    
    t0 = int(time.time())
    print('Execution start at', t0)
    print('Analyzing and PreProcessing the Data')
    ## transfering the feature "SKU_Code" to object and then viewing the result
    for df_name in df_li:
        if df_name not in key: continue
        print('"{}" dataframe shape:  {}'.format(df_name, df_li[df_name].shape))

        ## Changing feature data type to object
        df_li[df_name][['SKU_Code']] = df_li[df_name][['SKU_Code']].astype(str)

        ## Changing feature data type to datetime
        df_li[df_name]['Date'] = pd.to_datetime(df_li[df_name]['Date'],format='%Y-%m-%d') 

        ## sorting df based on Date
        df_li[df_name].sort_values(by=['Date'], inplace=True)
        df_li[df_name].reset_index(drop=True, inplace=True)

        # display(df_li[df_name].head())
        _ = datasetPrimAnalysis(df_li[df_name])
        print('****'*25,'\n\n')

    ## Loading Dataset to generate estimate on Quantity
    temp_sDF, temp_pDF, temp_iDF = df_li[key[0]].copy(), df_li[key[1]].copy(), df_li[key[2]].copy()
    temp_pDF = pd.DataFrame(temp_pDF.groupby(by= ['Store_Code', 'SKU_Code', 'Date'])['Qty'].sum().sort_index())#.reset_index()
    temp_sDF = pd.DataFrame(temp_sDF.groupby(by= ['Store_Code', 'SKU_Code', 'Date'])['Sales_Qty'].sum().sort_index())#.reset_index()
    # display(temp_pDF.head())
    # display(temp_sDF.head())
    print('Combining Secondary and Primary Datasets')
    totQua = temp_sDF.join(temp_pDF, how='outer').reset_index().fillna(0)
    print('DataFrame Shape', totQua.shape)
    # display(totQua.head())    
    
    
    ## Working with Smalller Data which liess between Jan2017 and Mar 2017
    t1 = int(time.time())
    print('TimeTaken {} sec\n'.format(t1-t0))
    print('Selecting the Data according to the mentioned Date Range ("2017-01-01" - "2017-03-31")')
    totQua = totQua.loc[(totQua['Date'] > '2017-01-01') & (totQua['Date'] <= '2017-03-31'), :]
    totQua.reset_index(drop=True, inplace=True)
    print('New DataFrame Shape', totQua.shape)
    
    
    t2 = int(time.time())
    print('TimeTaken {} sec\n'.format(t2-t1))
    print('Working on determining adequate initial value')
    ## Determining the adequate initial value
    totQua['QtyCn'] = totQua['Qty'].subtract(totQua['Sales_Qty'])
    adQn = totQua.groupby(by=['Store_Code', 'SKU_Code', 'Date'])['QtyCn'].sum()
    adQn = adQn.groupby(['Store_Code', 'SKU_Code']).cumsum() ## Cumulation over 'Storecode' & Sku
    cumQuant = adQn
    adQn = adQn.groupby(['Store_Code', 'SKU_Code']).min()
    totQua.drop(columns=['QtyCn'], inplace=True)

    ## Adding Closing Inventory 
    totQua['ClosingInvOfDayWhenBaseIsZero'] = list(cumQuant) ##0 ---> this value is not 

    ## Adding an assumed initial quantity
    templi = []
    for index, row in totQua.iterrows():
        val = adQn[row['Store_Code']][row['SKU_Code']] * -1
        templi.append(0 if val <=0  else val)
    totQua['BaseValToAdd'] = templi ## i.e. x in the series

    ## Adding The Final relavent Column
    totQua['LeastPossibleClosingInvOfDay'] = totQua['LeastPossibleClosingInvOfDay'] = totQua['ClosingInvOfDayWhenBaseIsZero'].add(totQua['BaseValToAdd'])

    ## Dropping Unnecessary Columns
    t3 = int(time.time())
    print('TimeTaken {} sec\n'.format(t3-t2))
    
    print('Dropping columns')
    totQua.drop(columns=['ClosingInvOfDayWhenBaseIsZero', 'BaseValToAdd'], inplace=True)

    '''
    ## Computing the overall Quantity -- Starting Qunatity is Computed such that qunatity was always available
    tSt = int(time.time())
    for st in totQua['Store_Code'].unique():
        for prod in totQua.loc[totQua['Store_Code']==st,'SKU_Code'].unique():
            t = totQua.loc[(totQua['Store_Code']==st) & (totQua['SKU_Code']==prod), \
                           ['Date', 'Sales_Qty', 'Qty', 'ClosingInvOfDay']].reset_index(drop=True)
            for i in t.index:
                totQua.loc[(totQua['Store_Code']==st) & (totQua['SKU_Code']==prod) & \
                           (totQua['Date'] == t['Date'][i]), 'ClosingInvOfDay'] = quantInStore
    print('Total Time Taken to compute  this {} sec'.format(int(time.time())-tSt))
    '''

    ## Resultant DataFrame --- Result Shown using groupby
    agg_Qty = totQua.groupby(by=['Store_Code', 'SKU_Code', 'Date']).sum().sort_index()
    # display(agg_Qty)

    t4 = int(time.time())
    print('TimeTaken {} sec\n'.format(t4-t3))
    ## Working with Smalller Data which liess between Jan2017 and Mar 2017
    print('Joining The Inventory Dataset')
    temp_iDF = temp_iDF.loc[(temp_iDF['Date'] > '2017-01-01') & (temp_iDF['Date'] <= '2017-03-31'), :]
    temp_iDF.reset_index(drop=True, inplace=True)

    temp_iDF = pd.DataFrame(temp_iDF.groupby(by= ['Store_Code', 'SKU_Code', 'Date'])['SOH'].sum().sort_index())#.reset_index()
    # display(temp_pDF.head())

    OverQua = agg_Qty.join(temp_iDF, how='outer').reset_index()#.fillna(0)

    ## observing the DF
    OverQua.groupby(by=['Store_Code', 'SKU_Code', 'Date']).first()
    
    t5 = int(time.time())
    print('TimeTaken {} sec\n'.format(t5-t4))
    print('Whole Execution Time {} sec\n'.format(t5-t0))

    return OverQua


if __name__ == "__main__":
    
    primDf = pd.read_csv('data/WC_DS_Ex1_Pri_Sales.csv')
    secDf = pd.read_csv('data/WC_DS_Ex1_Sec_Sales.csv')
    invDf = pd.read_csv('data/WC_DS_Ex1_Inv.csv')

    df_li = {'Secondary': secDf, 
             'Primary':primDf, 
             'Inventory': invDf
            }
    question4(df_li)