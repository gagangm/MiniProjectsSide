import matplotlib 
matplotlib.use('Agg')
## Loading Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef,accuracy_score
#from lib import datasetPrimAnalysis, ScalingDF, plotFeatureAndProperty, visualizeFeature, plotConfusionMatrix 
#from lib import plotAccAndErrorWrtThreshold, DimenRed_Visual, DetNoOfClusters

seed = 12345


## Doing Primary analysis
def getMissingPct(df):
    '''    '''
    return [ round((df['IsNullSum'][i] / max(df['count']))*100,2) for i in range(len(df)) ]

def genMsg(txt):
    '''    '''
    print('_'*12+'| Number of feature/s which are {} : {} |'.format(*txt)+'_'*12)

def datasetPrimAnalysis(DF, msg=True):
    '''
    Function which is used to analyze features and provide data insight
    '''
    df_explore = DF.copy()
    if msg: print('Overall dataset shape :', df_explore.shape)
    
    ## Creating a dataset that explain feature/s
    temp = pd.DataFrame(df_explore.isnull().sum(), columns = ['IsNullSum'])
    temp['dtypes'] = df_explore.dtypes.tolist()
    temp['IsNaSum'] = df_explore.isna().sum().tolist()
    
    ## Analyzing Time based Features
    temp_tim = temp.loc[temp['dtypes']=='datetime64[ns]' ,:]
    if (len(temp_tim) > 0):
        df_tim = df_explore.loc[:,temp_tim.index].fillna('Missing-NA')
        if msg: genMsg(['Time based', df_tim.shape[1]])
        temp_tim = temp_tim.join(df_tim.describe().T).fillna('')
        temp_tim['%Missing'] = getMissingPct(temp_tim)
        if msg: print(temp_tim)
    
    
    ## Analyzing Qualitative Features
    temp_cat = temp.loc[temp['dtypes']=='O' ,:]
    if (len(temp_cat) > 0):
        df_cat = df_explore.loc[:,temp_cat.index].fillna('Missing-NA')
        if msg: genMsg(['Qualitative', df_cat.shape[1]])
        temp_cat = temp_cat.join(df_cat.describe().T).fillna('')
        temp_cat['CategoriesName'] = [ list(df_cat[fea].unique()) for fea in temp_cat.index ]
        temp_cat['%Missing'] = getMissingPct(temp_cat)
        if msg: print(temp_cat)
    
    
    ## Analyzing Quantitative Features
    temp_num = temp.loc[((temp['dtypes']=='int') | (temp['dtypes']=='float')),:]
    if (len(temp_num) > 0):
        df_num = df_explore.loc[:,temp_num.index]#.fillna('Missing-NA')
        if msg: genMsg(['Quantitative', df_num.shape[1]])
        temp_num = temp_num.join(df_num.describe().T).fillna('')
        temp_num['%Missing'] = getMissingPct(temp_num)
        if msg: print(temp_num)
    # if temp_cat['dtypes'][i] == 'float', 'int', 'O'

    if len(temp)!= len(temp_tim) + len(temp_cat) + len(temp_num):
        print("Some columns data is missing b/c of data type")
    
    dit = {'TimeBased': temp_tim, 'Categorical': temp_cat, 'Numerical': temp_num}
    return dit

## Defining Scaling DF
class ScalingDF:
    '''
    This class can be used for scaling features. 
    For the Train Cycle, feat_info_dict (i.e. information aboout the features) shall be 'None' or undefined 
    For the Predict Cycle, feat_info_dict (i.e. information aboout the features) MUST be provided.
        - This feat_info_dict is obtained ad an additional argument the some scaling is done
        - Can again be obtained using 'getInitialFeaturesDescriptiveStats' for the same instance
    
    "getInitialFeaturesDescriptiveStats" provide descriptive information on the in DF on which 'ScalingDF' 
                                        was initialized
    "generateNewFeaturesDescriptiveStats" provides descriptive information on the features after they have 
                                        been tranformed byy any method
    '''
    def __init__(self, df, feat_info_dict = None):
        
        df = df.copy()
        if feat_info_dict is None:
            ## Computing Measures used for Scaling
            feat_info_dict = {}
            for col in df.columns:
                feat_info_dict[col] = {'Min': df[col].min(),
                                       'Median': df[col].median(), 
                                       'Max': df[col].max(), 
                                       'Mean': df[col].mean(), 
                                       'Std': df[col].std()}
        else:
            ## Check if columns are matching if nnot raise ann error
            colNotPresent = len([ False for ele in feat_info_dict.keys() if ele not in df.columns ])
            if colNotPresent > 0:
                raise Exception('Feature that is to be scaled is not present in the provided DF')
        
        self.df = df
        self.feat_info_dict = feat_info_dict
    
    def getInitialFeaturesDescriptiveStats(self):
        return self.feat_info_dict
    def generateNewFeaturesDescriptiveStats(self):
        feat_info_dict = {}
        for col in self.df.columns:
            feat_info_dict[col] = {'Min': self.df[col].min(),
                                   'Median': self.df[col].median(), 
                                   'Max': self.df[col].max(), 
                                   'Mean': self.df[col].mean(), 
                                   'Std': self.df[col].std()}
        return feat_info_dict
        
    def normalization(self):
        print('Scaling dataframe using {} scaler'.format('Normalization'))
        for col in self.df.columns:
            print('|\t', col)
            li = list(self.df[col])
            self.df[col] = [ (elem - self.feat_info_dict[col]['Min']) / \
                            (self.feat_info_dict[col]['Max'] - self.feat_info_dict[col]['Min']) \
                            for elem in li ] 
        return self.df, self.feat_info_dict
    
    def standardization(self):
        print('Scaling dataframe using {} scaler'.format('Standardization'))
        for col in self.df.columns:
            print('|\t', col)
            li = list(self.df[col])
            self.df[col] = [ (elem - self.feat_info_dict[col]['Mean']) / self.feat_info_dict[col]['Std']\
                            for elem in li ]
        return self.df, self.feat_info_dict
    
    def standard_median(self):
        print('Scaling dataframe using {} scaler'.format('Standard_Median'))
        for col in self.df.columns:
            print('|\t', col)
            li = list(self.df[col])
            self.df[col] = [ (elem - self.feat_info_dict[col]['Median']) / self.feat_info_dict[col]['Std'] \
                            for elem in li ] 
        return self.df, self.feat_info_dict



def dataCleaningFunction(df, predictInfo=None):
    df = df.copy()
    cycle = 'Train' if predictInfo == None else 'Predict'
    print('\n', cycle)
    
    ## datetime based feature
    feat_dt = ['F15', 'F16']
    for f in feat_dt: df[f] = pd.to_datetime(df[f],format='%m/%d/%Y')
    ## Categorical Features
    feat_str = ['Index']
    feat_str += ['F5', 'F6', 'F7', 'F8', 'F9']
    feat_str += ['F17', 'F18']
    feat_str += ['F19', 'F20']
    feat_str += ['F21', 'F22']
    if cycle == 'Train': feat_str += ['C']
    # Converting feature datatype to Object
    for f in feat_str: df[f] = df[f].astype(str)
    #df_info = datasetPrimAnalysis(df)

    ## Converting Information Type
    df['F15_d'] = [ ele.timestamp() / (60*60*24) for ele  in df['F15'] ]
    df['F16_d'] = [ ele.timestamp() / (60*60*24) for ele  in df['F16'] ] 
    # To Numeric
    feat_num = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12',\
                'F13','F14','F15_d','F16_d','F19','F20','F21','F22']
    for f in feat_num: df[f] = df[f].astype(float)
    # To Object
    if cycle == 'Train': 
        feat_str = ['Index','F17','F18','C']
    else:
        feat_str = ['Index','F17','F18']
    for f in feat_str: df[f] = df[f].astype(str)
    ## Dropping some certain features
    df.drop(columns=['F15', 'F16'], inplace=True) #'Index' is valuable for sub_df hence won't be dropped at this stage
    df_info = datasetPrimAnalysis(df, msg=False)
    
    if cycle == 'Train': df.drop(columns=['Index'], inplace=True)
    
    ## Handling Missing Obsservation
    '''Nothing to Do'''
    
    ## Feature Scaling
    if cycle == 'Train':
        predictInfo = {}
        scaler = ScalingDF(df.loc[:, df_info['Numerical'].index ])
        df.loc[:, df_info['Numerical'].index ], predictInfo['scalerDict'] = scaler.standardization()
    else:
        scaler = ScalingDF(df.loc[:, df_info['Numerical'].index ], predictInfo['scalerDict'])
        df.loc[:, df_info['Numerical'].index ], iniDescStats_dict = scaler.standardization()
    df_info = datasetPrimAnalysis(df, msg=False)    
    
    ## Dummy Feature Creation
    colToTransform = ['F17', 'F18']
    df = df.join(pd.get_dummies(df.loc[:, colToTransform], prefix=None, prefix_sep='_'))
    df.drop(columns=colToTransform, inplace=True)
    
    ## Handling Outlier
    '''Nothing to Do'''
    
    ## Class Balance & Splitting Dataset
    '''Nothing to Do'''
    
    return df, predictInfo


def plotAccAndErrorWrtThreshold(yActual, yPredictedStocastic):
    ''' . '''
    ResultDF, i, j, step, ClassThres = pd.DataFrame(), 0.0, 1.0, 0.01, []
    while i <= j: ClassThres.append(round(i,3)); i += step

    i = 0
    for limit in ClassThres:
        predicted = np.copy(yPredictedStocastic)
        ResultDF.loc[i, 'ThresholdValue'] = limit
        predicted[predicted > limit] = 1
        predicted[predicted <= limit] = 0
        ResultDF.loc[i, 'Accuracy'] = accuracy_score(yActual, predicted)
        ResultDF.loc[i, 'ErrorRate'] = 1 - ResultDF.loc[i, 'Accuracy']
        ResultDF.loc[i,'MCC'] = matthews_corrcoef(yActual, predicted)
        i += 1
    ser_thres, ser_acc, ser_err = ResultDF['ThresholdValue'], ResultDF['Accuracy'], ResultDF['ErrorRate']
    
    fig = plt.figure(figsize=(15,6))
    ax = fig.gca()
    
    plt.subplot(121)
    ax.set_xticks(np.arange(start = 0, stop = 1.01, step = 0.1))
    ax.set_yticks(np.arange(start = 0, stop = 100.01, step = 10))
    plt.plot(ser_thres, 100*ser_acc, label = 'Accuracy', lw=3)
    plt.plot(ser_thres, 100*ser_err, label = 'Error Rate', lw=3)
    plt.title('Evaluation Parameters VS Threshold', fontsize=15)
    plt.xlabel('Threshold Value', fontsize=13)
    plt.ylabel('Percentage', fontsize=13)
    plt.legend(fontsize = 13)
    #plt.xticks(np.arange(start = 0, stop = 1.01, step = 0.1))
    #plt.yticks(np.arange(start = 0, stop = 100.01, step = 10))
    plt.axis([0,1,0,100])
    plt.axhline(0, color='black',lw=3)
    plt.axvline(0, color='black',lw=3)
    plt.margins(1,1)
    plt.grid(True, color = 'black', alpha = 0.3)
    #plt.label()
    #plt.rc
    
    maxMCC = max(ResultDF['MCC'])
    threshMaxMCC = list(ResultDF.loc[ResultDF['MCC']==maxMCC, 'ThresholdValue'])[0]
    
    plt.subplot(122)
    ax.set_xticks(np.arange(start = 0, stop = 1.01, step = 0.1))
    ax.set_yticks(np.arange(start = 0, stop = 100.01, step = 10))
    plt.plot(ser_thres, ResultDF['MCC'], label = 'MCC', lw=3)
    plt.title('Matthew Correlation Coefficient VS Threshold', fontsize=15)
    plt.xlabel('Threshold Value', fontsize=13)
#     plt.ylabel('Percentage', fontsize=13)
    plt.legend(fontsize = 13)
    #plt.xticks(np.arange(start = 0, stop = 1.01, step = 0.1))
    #plt.yticks(np.arange(start = 0, stop = 100.01, step = 10))
    plt.axis([0,1,0,1])
    plt.axhline(maxMCC, color='black',lw=1.5)
    plt.axvline(threshMaxMCC, color='black',lw=1.5)
    plt.margins(1,1)
    plt.grid(True, color = 'black', alpha = 0.3)
    
    plt.show()
    return threshMaxMCC
# threshold = plotAccAndErrorWrtThreshold(ytest, ypred) 

def applyXGB(xtrain, ytrain, xtest, ytest, xSubDF):
    '''
    - Computing best features
    - Using Best Fetures to Train a Model
    - Generate a result file
    '''
    xtrain, ytrain, xSubDF = xtrain.copy(), ytrain.copy(), xSubDF.copy()
    ind_params = {
        'objective': 'binary:logistic'
        }
    cv_params = {
        'learning_rate': [0.1, 0.2],
        'max_depth': [4, 5, 6],
        'subsample': [1],
        'colsample_bytree': [1]
        }
    # Create a based model
    xgbC = xgb.XGBClassifier(**ind_params)
    # Instantiate the grid search model
    grid_search_xgb = GridSearchCV(estimator = xgbC, param_grid = cv_params, scoring='roc_auc', n_jobs=-1, verbose=1)
    # Fit the grid search to the data
    grid_search_xgb.fit(xtrain, ytrain)
    
    # Best params
    best_parameters_xgb, score_xgb, _ = max(grid_search_xgb.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score_xgb)
    for param_name in sorted(best_parameters_xgb.keys()):
        print("%s: %r" % (param_name, best_parameters_xgb[param_name]))
    print('\nBest Parameter Combination: \n|\t',grid_search_xgb.best_params_)
    
    ## Training the Model
    params = best_parameters_xgb ## rest can be default
    xgbC = xgb.XGBClassifier(**params, random_state=seed, n_jobs=-1)
    xgbC.fit(xtrain, ytrain)
    
    ## Testing the Model
    # Deterministic Prediction 
    # ypred = xgbC.predict(xtest) 
    # plotConfusionMatrix(y_act=ytest, y_pred=ypred)
    # Stochastic Prediction 
    ypred = xgbC.predict_proba(xtest)
    ypred = [ ypred[i][1] for i in range(len(ypred)) ]

    # Understanding How Accuracy Changes via changing threshold
    threshold_best_mcc = plotAccAndErrorWrtThreshold(ytest, ypred) 
    print('Best Predictive power of the model is when the threshold is around {}'.format(threshold_best_mcc))
    print('Though the accuracy will be low at this threshold ~ 55%')
    
    ## Generating Result for other data
    threshold = 0.27
    xSubDF_Fil = xSubDF.loc[:,[col for col in xSubDF.columns.values if col != 'Index']]
    ypred = xgbC.predict_proba(xSubDF_Fil)
    ypred = [ ypred[i][1] for i in range(len(ypred)) ]
    xSubDF.loc[:,'Class'] = ypred
    xSubDF.loc[:,['Index', 'Class']].to_csv('result_xgb_probalistic_pred_Class.csv', sep='\t', index=False)
    xSubDF.loc[:,'C'] = [ 1 if ele > threshold else 0 for ele in ypred]
    xSubDF.loc[:,['Index', 'Class']].to_csv('result_xgb_deterministic_pred_Class.csv', sep='\t', index=False)

def applyNaiveBayes(xtrain, ytrain, xtest, ytest, xSubDF):
    '''
    '''
    xtrain, ytrain, xSubDF = xtrain.copy(), ytrain.copy(), xSubDF.copy()
    
    nb = GaussianNB()
    nb.fit(xtrain, ytrain)
    
    ## Generating Result for other data
    xSubDF_Fil = xSubDF.loc[:,[col for col in xSubDF.columns.values if col != 'Index']]
    xSubDF.loc[:,'Class'] = nb.predict(xSubDF_Fil)
    xSubDF.loc[:,['Index', 'Class']].to_csv('result_naiveBayes_deterministic_pred_Class.csv', sep='\t', index=False)

def main():
    '''
    matthew correlation coefficient was used as the primary indicator for the model performace estimation
    '''
    ## Loading Data
    # Train Data
    df = pd.read_csv('ClassificationProblem1.txt', delimiter='\t')
    print('DataFrame shape is {}'.format(df.shape))
    print(df.head())
    # Submission Data
    sub_df = pd.read_csv('Classification1Test.txt', delimiter='\t')
    print('\nDataFrame shape is {}'.format(sub_df.shape))
    print(sub_df.head())
    
    ## Data Preprocessing
    cleanDF, predictInfo = dataCleaningFunction(df)
    xSubDF, predictInfo = dataCleaningFunction(sub_df, predictInfo=predictInfo)
    
    # Splitting the dataset into the Training set and Test set
    ## Dividing the data into X and Y
    x = cleanDF.loc[:, cleanDF.columns != 'C'].copy()
    y = cleanDF['C']
    ## Changing Datatype of Critical Class
    y = y.astype(int)
    ## Splitting the dataset
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)
    print('length of train and test set :', len(xtrain), len(xtest) )
    
    ## Apply XGBoost
    applyXGB(xtrain, ytrain, xtest, ytest, xSubDF)
    
    ## Apply NaiveBayes
    applyNaiveBayes(xtrain, ytrain, xtest, ytest, xSubDF)

if __name__ == "__main__":
    main()
