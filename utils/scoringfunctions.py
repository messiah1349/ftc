import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
import matplotlib
import matplotlib.pyplot as plt
from math import log
from sklearn import metrics as met
from sklearn.ensemble import ExtraTreesClassifier
from pylab import rcParams
from sklearn.utils import check_consistent_length, column_or_1d, check_array
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import seaborn as sns
import woe
import ipywidgets as widgets
import os
import getpass
    
def colCnt(df):
    d = []
    for col in df.columns:
        d.append((col,len(df[col].unique())))  

    colCntDescr = pd.DataFrame(d,columns=['col','uinqValuesCount'])
    return colCntDescr
    
def getCategoricalColumn(df,column,clustNum):
    df = df.reset_index(drop=True)
    
    if df[column].min() == 0:                                     #если есть 0, то его берем в отдельный кластер
        dfColNotZero = df.loc[df[column]>0,column]
        bins = pd.core.algorithms.quantile(np.unique(dfColNotZero),np.linspace(0, 1, clustNum))
        bins = np.insert(bins,0,0)
    else:
        bins = pd.core.algorithms.quantile(np.unique(df[column]),np.linspace(0, 1, clustNum+1))
    
    result = pd.cut(df[column], bins, include_lowest=True)   # колонка в виде категорий
    categoricalTable = pd.DataFrame(list(zip(df[column],result)),columns=[column,'categorical']) # сцепка исходной колонки с категориями
    newColumnName = column + '_cat'     #имя новой колонки
    grouped = categoricalTable.groupby('categorical')
    minCatValue = grouped.min().reset_index().rename(columns={column:newColumnName}) 
    categoricalTable = pd.merge(categoricalTable,minCatValue,how='left') #добавляем минимальное значение в категории, это будет имя нашего кластера
    dfr = df.copy()
    dfr[newColumnName] = categoricalTable[newColumnName]  #исходный датафрейм с новой колонкой 
    minCatValue['maxVal'] = grouped.max()[column].values
    minCatValue.rename(columns = {newColumnName: 'minVal'},inplace=True)
    return dfr, minCatValue

def getClustColumnTransform(x,minVal,maxVal):
    if np.isnan(x):
        return np.nan
    for mn, mx in zip(minVal,maxVal):
        if x <= mx:
            return mn
    return minVal[-1]
        
def getClustTransform(df,clustVarsInfo):
    dfOut = df.copy()
    clustVars = clustVarsInfo.variable.unique()
    for var in clustVars:
        varTransformTable = clustVarsInfo[clustVarsInfo.variable==var].sort_values('maxVal')
        newColumnName = var + '_cat'
        dfOut[newColumnName]=dfOut[var].apply(getClustColumnTransform,args=[list(varTransformTable.minVal),list(varTransformTable.maxVal)])
    dfOut = dfOut.drop(clustVars,axis=1)
    return dfOut

def getWOETransform(df,woeVarsInfo):
    dfOut = df.copy()
    woeVars = woeVarsInfo.variable.unique()
    for var in woeVars:
        varTransformTable = woeVarsInfo[woeVarsInfo.variable==var]
        newColumnName = var + '_WOE'
        dfOut[newColumnName]=dfOut[var].apply(getWOEColumnTransform,args=[list(varTransformTable.maxVal),list(varTransformTable.WOE)])
    dfOut = dfOut.drop(woeVars,axis=1)
    return dfOut

def getWOEColumnTransform(x,maxVal,WOE):
    if np.isnan(x) & np.isnan(maxVal[0]):
        return WOE[0]
    for mx, w in zip(maxVal,WOE):
        if x <= mx:
            return w
    return WOE[-1]

def continuousVariables(df,columnLimit=50):
    colCntDescr = colCnt(df)
    
    clustVarsInfo = pd.DataFrame(columns = ['categorical','minVal','maxVal','variable'])

    dfPreWoe = df.copy()

    continuousVars = colCntDescr.loc[colCntDescr['uinqValuesCount']>columnLimit,'col'].values

    for col in continuousVars:
        dfPreWoe, clusColInfo = getCategoricalColumn(dfPreWoe,col,columnLimit)
        clusColInfo['variable'] = col
        clustVarsInfo = pd.concat([clustVarsInfo,clusColInfo])

    dfPreWoe = dfPreWoe.drop(continuousVars,axis = 1 )
    
    return dfPreWoe, clustVarsInfo

def woeVariables(df,badFlag,rateLimit=0.05,minBadRateDiff=0.01,minBins=2,maxBins=5,columns=None,badFlag2 = None, show_print=False):
    if columns is not None:
        colCntDescrPreWoe = colCnt(df[columns])
    else:
        colCntDescrPreWoe = colCnt(df)
    colCntDescrPreWoe = colCntDescrPreWoe[~colCntDescrPreWoe['col'].isin([badFlag,badFlag2])]
    woeColumns = list(colCntDescrPreWoe.loc[colCntDescrPreWoe['uinqValuesCount']>1,'col'])
    
    woeVarsInfo = pd.DataFrame()#(columns = ['minVal', 'maxVal', 'bads', 'total', 'goods', 'badRate', 'goodRate','WOE','variable'])

    dfPostWoe = df.copy()
    
    colPos = 0.0
    totalCols = len(woeColumns)
    if show_print: print('Progress: ',end="")
    for woeColumn in woeColumns:
        if show_print: print(woeColumn,end="")
        dfPostWoe, woeColInfo = woe.getWOEcolumn(dfPostWoe,woeColumn,badFlag,rateLimit,minBadRateDiff,minBins,maxBins,badFlag2) 
        woeColInfo['variable'] = woeColumn
        woeVarsInfo = pd.concat([woeVarsInfo,woeColInfo])
        colPos+=1.0
        
        if show_print: print("=%.1f%%, " % (colPos/totalCols * 100),end="")
    dfPostWoe = dfPostWoe.drop(woeColumns,axis=1)
    if show_print: print("")
    return dfPostWoe, woeVarsInfo

def getWOEcolumnAfterTransform(dfS,woeVarsInfo):
    df = dfS.copy()
    WOEvariables = list(woeVarsInfo.variable.unique())
    for var in WOEvariables:
        #print(var)
        WOEvarsInfoCur = woeVarsInfo[woeVarsInfo.variable==var]
        maxValv = WOEvarsInfoCur.maxVal.values
        WOEv = WOEvarsInfoCur.WOE.values
        newColumn = var + '_WOE'
        df[newColumn] = df[var].apply(woe.getWOE,args=[maxValv,WOEv])
    df = df.drop(WOEvariables,axis=1)
    return df

def corrTable(df,informationTable):
    CorrKoef = df.corr()
    corColumns = CorrKoef.columns
    c1 = []
    c2 = []
    corVal = []
    for i in range(1,len(CorrKoef)):
        for j in range(i+1,len(CorrKoef)):
            if CorrKoef.iloc[i,j]>0.5:
                c1.append(corColumns[i])
                c2.append(corColumns[j])
                corVal.append(CorrKoef.iloc[i,j])
    corDf = pd.DataFrame({'var1':c1,'var2':c2,'r^2':corVal}).reindex_axis(['var1','var2','r^2'],axis=1).sort_values('r^2',ascending=False)
    if len(corDf) == 0: return corDf,0
    allCorVars = list(corDf.var1) + list(corDf.var2)
    dfO = pd.DataFrame({'variable':allCorVars})
    dfOg = dfO.groupby('variable').size().reset_index().sort_values(0,ascending=False)
    dfOg = pd.merge(dfOg,informationTable[['variable','InformationValue']])
    dfOg = dfOg.rename(columns = {0:'varCorrelationCount'})
    return corDf,dfOg

def getIVfromWOE(woeDf):
    ivs = []
    variables = woeDf.variable.unique()
    for var in variables:
        vardf = woeDf[woeDf.variable==var]
        iv = -((vardf.goodRate - vardf.badRate) * vardf.WOE).sum() / 100
        ivs.append(iv)
    dfOut = pd.DataFrame(list(zip(variables,ivs)),columns = ['variable','InformationValue'])
    dfOut = dfOut.sort_values('InformationValue',ascending=False)
    return dfOut

def preClean(x):
    if x[-4:]!='_WOE': a = x
    else: a = x[:-4]
        
    if a[-4:]!='_cat': return a
    else: return a[:-4]
   

def columnClean(columns):
    cwoCAT = list(map(preClean,columns))
    return cwoCAT

def ootTransform(ootDf,clustVarsInfo,woeVarsInfo,goodColumns):
    
    
    if len(clustVarsInfo)==0:
        clustVarsInfo = pd.DataFrame(columns = ['categorical','minVal','maxVal','variable'])
    
    columnCleans = columnClean(goodColumns)
    clust = clustVarsInfo[clustVarsInfo.variable.isin(columnCleans)]
    woe = woeVarsInfo[woeVarsInfo.variable.apply(preClean).isin(columnCleans)]
    
    dfOotPreWOE = getClustTransform(ootDf,clust)
    dfOotPostWOE = getWOETransform(dfOotPreWOE,woe)
 
    
    d = dict(zip(list(dfOotPostWOE.columns),columnClean(dfOotPostWOE.columns)))
    outpColumns = [k for k,v in d.items() if v in columnCleans]
     
    dfFPDpreLR = dfOotPostWOE[outpColumns]
    return dfFPDpreLR

def featureImportance(X,y,columns):
    model = ExtraTreesClassifier()
    model.fit(X,y)
    fi = pd.DataFrame(np.array([columns,model.feature_importances_]).T,columns=['variable','feature_importance']).sort_values('feature_importance',ascending = False)
    return fi

def giniGrowth(df,woeVarsInfo,badFlag,iv_limit=0.015, random_state=3):
    woeTable = woeVarsInfo.copy()
    woeTable.variable = woeTable.variable.apply(lambda x: x + '_WOE')
    IV = getIVfromWOE(woeTable)
    columns = IV[IV.InformationValue>iv_limit].variable
    infoValue = list(IV[IV.InformationValue>iv_limit].InformationValue)
    columnsForModeking = []
    giniTest = []
    giniTrain = []
    y = df[badFlag].values
    for col in columns:
        columnsForModeking.append(col)
        X = df[columnsForModeking].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
        lr = LogisticRegression(solver='liblinear')
        lr.fit(X_train,y_train)
        pr_test = lr.predict_proba(X_test)[:,1]
        pr_train = lr.predict_proba(X_train)[:,1]
        rocGiniTest =  met.roc_auc_score(y_test,pr_test) * 2 - 1
        rocGiniTrain =  met.roc_auc_score(y_train,pr_train) * 2 - 1
        giniTest.append(rocGiniTest)
        giniTrain.append(rocGiniTrain)
    trainDiff = [x-y for x,y in zip(giniTrain,[0]+giniTrain[:-1])]
    testDiff = [x-y for x,y in zip(giniTest,[0]+giniTest[:-1])]
    dfOut = pd.DataFrame({'variable':columns, 'giniTrain' : giniTrain,'giniTest': giniTest,'trainDiff':trainDiff,'testDiff':testDiff,'informationValue':infoValue})
    dfOut[['trainDiff','testDiff']] = dfOut[['trainDiff','testDiff']]#.apply('${:,.2f}'.format)
    dfOut = dfOut.reindex_axis(['variable','informationValue','testDiff','trainDiff','giniTest','giniTrain'],axis=1)
    return dfOut

def information_table_filt(information_table,good_columns=None): 
    information_table_filt = information_table.copy()
    if good_columns is None:
        good_columns = list(information_table_filt.variable)
    good_columns_clean = columnClean(good_columns)
    information_table_filt.variable = information_table_filt.variable.apply(preClean)
    information_table_filt = information_table_filt[information_table_filt.variable.isin(good_columns_clean)]    
    return information_table_filt

def generate_woe_info_filt(woe_info, good_columns):
    woe_info_filt = woe_info[woe_info.variable.apply(preClean).isin(columnClean(good_columns))]
    return woe_info_filt

def chooseColumnsFromIT(informationTable,badFlag = 'badMob3', min_limit = 0.002):
    goodColumns= list(informationTable[(informationTable['testDiff']>min_limit)&(informationTable['trainDiff']>min_limit)].variable.values) + [badFlag]
    badColumns = list(informationTable[(informationTable['testDiff']<=min_limit)|(informationTable['trainDiff']<=min_limit)].variable.values) 
    return goodColumns, badColumns

def woeOutput(woeInfoTrans,goodColumns,model,factor):
    woeInfoOutput = woeInfoTrans.copy()
    goodC = [x[:-4] for x in model.columns]
    d = dict(zip(goodC,model.lr.coef_.tolist()[0]))
    woeInfoOutput = woeInfoOutput[woeInfoOutput['variable'].isin(list(d.keys()))]
    woeInfoOutput['coef'] = ''
    for var in d:
        woeInfoOutput.loc[woeInfoOutput['variable']==var,'coef']=d[var]
    woeInfoOutput['scorValue'] = -woeInfoOutput['WOE']*woeInfoOutput['coef']*factor    
    woeInfoOutput = woeInfoOutput[['variable','minVal','maxVal','scorValue','coef','WOE']]
    return woeInfoOutput

def woeProduction(woeOutp):
    woeOut = woeOutp.copy()
    woeOut['varInit'] = woeOut.variable.apply(lambda x: x if x[-4:]!='_cat' else x[:-4] )
    variables = list(woeOut.varInit.unique())
    for var in variables:
        woeOut.loc[(woeOut.varInit==var)&(woeOut.maxVal.notnull()),
                   'maxVal'] = list(woeOut.loc[(woeOut.varInit==var)&(woeOut.maxVal.notnull()),'minVal'])[1:] + [100000000]
    woeOut = woeOut[['varInit','maxVal','scorValue']]
    return woeOut

def variableToScoringValue(x,mv,sv):
    ######mv = wouOutpVar.minVal.values
    #sv = wouOutpVar.scorValue.values
    if np.isnan(mv[0]):
        svT = sv[1:]
        mvT = np.concatenate([mv[2:],[1000000000]])
    else:
        svT = sv
        mvT = np.concatenate([mv[1:],[1000000000]])
    if np.isnan(x):
        return sv[0]
    else:
        l = len(np.where(mvT<=x)[0])
        return svT[l]

def getScoringTable(dfSrc,woeOut,intercept,IdColumn):
    df = dfSrc.copy()
    woeOut['varInit'] = woeOut.variable.apply(preClean)
    variables = list(woeOut['varInit'].unique())
    weightsDf = pd.DataFrame(np.zeros((len(df),len(variables))),columns = variables)
    
    for var in variables:
        
        woeVar = woeOut[woeOut.varInit==var]
        mv = woeVar.minVal.values
        sv = woeVar.scorValue.values
        weightsDf[var] = df[var].apply(variableToScoringValue,args=[mv,sv]).values
    
    weightsDf['scoring'] = weightsDf.sum(axis = 1) + intercept 
    weightsDf[IdColumn] = df[IdColumn].values
    weightsDf = weightsDf.reindex_axis([IdColumn,'scoring']+variables, axis=1)
    
    return weightsDf

def writeScoringTable(folderName,scoringTable,appLoans,rewrite=False):
    folderNameFull = folderName + 'modelDescription'
    if not os.path.exists(folderNameFull):
        os.makedirs(folderNameFull)
    
    if os.path.isfile(folderNameFull + '/scoringTable.csv') and not rewrite:
        print('File Have Already Existed')
        return
    
    scoringTable['isReal'] = 0
    scoringTable.loc[scoringTable.clientKey.isin(appLoans.clientKey),'isReal'] = 1
    #scoringTable[['clientKey','scoring','isReal']].to_csv(folderNameFull + '/scoringTable.csv', index=False)
    scoringTable.to_csv(folderNameFull + '/scoringTable.csv', index=False)
    
def bucketRate(y_true,y_score,buckets=10):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    ysz = y_true.size
    ysm = y_true.sum()
    desc_score_indices = np.argsort(y_score, kind="mergesort") #[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    ixsSec = [int(ysz*(i/buckets))-1 for i in range(1,buckets+1)]
    ixsFirst = [0] + ixsSec[:-1]
    cummBadRate = []
    curBadRate = []
    cutOff = []
    for i in range(buckets):
        cummBadRate.append(y_true[:ixsSec[i]].mean())
        curBadRate.append(y_true[ixsFirst[i]:ixsSec[i]].mean())
        cutOff.append(y_score[ixsSec[i]])
    
    arBorders = np.arange(0, 1, 1/buckets) + 1/buckets
    arBorders = list(map(lambda x: round(x,0), list(arBorders*100)) )
    arBordersCur = list(map(lambda x,y: str(int(x)) + "-" + str(int(y)), [0]+arBorders[:-1], arBorders))
    cummBadRate = list(map(lambda x: round(x*100,1) ,cummBadRate))
    curBadRate = list(map(lambda x: round(x*100,1) ,curBadRate))
    
    df = pd.DataFrame({'AR_cum':arBorders, 'cumBadRate':cummBadRate,
                       'AR_cur': arBordersCur, 'curBadRate':curBadRate,
                      'CutOff': cutOff}).sort_values('AR_cum'
                    , ascending=False).reindex_axis(['AR_cum','cumBadRate','AR_cur','curBadRate','CutOff'], axis=1)
    return  df

def decAR(x,brr):
    l = len(brr)
    for i in range(1,l+1):
        if x <= brr[l-i]:
            return float(i) * 100 / l

def rocAuc(y_true,y_score):
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    fps, tps, _ = met.roc_curve(y_true,y_score, pos_label=1)
    rocGini =  met.roc_auc_score(y_true,y_score) * 2 - 1
    return fps, tps, rocGini

def rocCurve(y_true,y_score, to_file=False, folder=False, mark=None):
    
    fps, tps, rocGini = rocAuc(y_true,y_score)

    plt.figure()
    plt.plot(fps, tps, label='ROC curve (rocGini = %.4f)' % (rocGini))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if to_file:
        plt.savefig(folder + '/' + mark + '_rocAuc.png')
    else:
        plt.show()

def lorenzCurve(y_true,y_score):
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    fps, tps, lorenceGini = lorenzAuc(y_true,y_score)
    badRate = y_true.sum() / y_true.size
    plt.figure()
    plt.plot(fps, tps, label='Lorenz curve (gini = %.4f)' % (lorenceGini))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0, badRate], [0, 1], 'k--', c='green')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Lorenz Curve')
    plt.legend(loc="lower right")
    plt.show()

class logReg():
    def __init__(self,preLR,declDf=None,badFlag='badMob3', test_size=0.33,random_state=3):
        if declDf is None:
            y = preLR[badFlag]
            X = preLR.drop(badFlag,axis=1).values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=random_state)
            lr = LogisticRegression(C=0.1)
            lr.fit(X_train,y_train)

            self.lr = lr
            self.pr_test = lr.predict_proba(X_test)[:,1]
            self.pr_train = lr.predict_proba(X_train)[:,1]
            self.pr_total = lr.predict_proba(X)[:,1]
            self.y_test = np.array(y_test)
            self.y_train = np.array(y_train)
            self.y_total = np.array(y)
            self.columns = list(preLR.drop(badFlag,axis=1).columns)
            self.reject = False
        else:
            appDf = preLR.copy()
            declDf = declDf[list(appDf.columns)]
            appDeclDf = pd.concat([declDf,appDf])

            X_appDecl = appDeclDf.drop(badFlag,axis=1).values
            X_app = appDf.drop(badFlag,axis=1).values
            X_decl = declDf.drop(badFlag,axis=1).values
            y_appDecl = appDeclDf[badFlag].values
            y_app = appDf[badFlag].values
            y_decl = declDf[badFlag].values
            lr = LogisticRegression(C=0.1)
            lr.fit(X_appDecl,y_appDecl)

            self.lr = lr
            self.pr_test = lr.predict_proba(X_decl)[:,1]
            self.pr_train = lr.predict_proba(X_app)[:,1]
            self.pr_total = lr.predict_proba(X_appDecl)[:,1]
            self.y_test = np.array(y_decl)
            self.y_train = np.array(y_app)
            self.y_total = np.array(y_appDecl)
            self.columns = list(appDeclDf.drop(badFlag,axis=1).columns)
            self.reject = True
        
    def calc_gini(self):
        _a,_b,gini_train = rocAuc(self.y_train,self.pr_train)
        _a,_b,gini_test = rocAuc(self.y_test,self.pr_test)
        _a,_b,gini_total = rocAuc(self.y_total,self.pr_total)
        return gini_train, gini_test, gini_total
    
    def print_gini(self,reject=False):
        gini_train, gini_test, gini_total = self.calc_gini()
        if not self.reject:            
            print("="*62)
            print('giniTrain = %.4f'.center(60,"-") % gini_train)
            print(' giniTest = %.4f'.center(60,"-") % gini_test)
            print("="*62)
        else:
            print("="*62)
            print('----giniAll = %.4f'.center(60,"-") % gini_total)
            print('giniApprove = %.4f'.center(60,"-") % gini_train)
            print(' giniDecline = %.4f'.center(60,"-") % gini_test)
            print("="*62)
            
    def print_lorenz_curve(self):
        if not self.reject:
            print('TRAIN'.center(60,"="))
            print('Sample Size = %s' % len(self.y_train))
            lorenzCurve(self.y_train,self.pr_train)
            print('='*60)
            print('TEST'.center(60,"="))
            print('Sample Size = %s' % len(self.y_test))
            lorenzCurve(self.y_test,self.pr_test)
            print('='*60)
        else:
            print('All'.center(60,"="))
            lorenzCurve(self.y_total,self.pr_total)
            print('='*60)
            print('APPROVE'.center(60,"="))
            lorenzCurve(self.y_train,self.pr_train)
            print('='*60)
            print('DECLINE'.center(60,"="))
            lorenzCurve(self.y_test,self.pr_test)
            print('='*60)
        
    def print_roc_curve(self, to_file=False, folder=False):
        rocCurve(self.y_train,self.pr_train, to_file, folder, 'train')
        rocCurve(self.y_test,self.pr_test, to_file, folder, 'test')    
    
    def app_decline_sep(self,binSize):
        df_r_decl = pd.DataFrame({'pr':self.pr_test,'y':self.y_test})
        df_r_decl['app'] = 'decline'
        df_r_app = pd.DataFrame({'pr':self.pr_train,'y':self.y_train})
        df_r_app['app'] = 'approve'
        df_r = pd.concat([df_r_decl,df_r_app]) 
        df_r = df_r.sort_values('pr').reset_index(drop=True)
        df_r['buck'] = 0
        sh = len(df_r)//binSize
        for i in range(binSize):
            df_r.loc[sh*i:sh*(i+1),'buck'] = i*(100//binSize)

        g = sns.factorplot(x="buck",  hue="app", data=df_r, kind="count", palette="muted", size=6, color = ['r','b'])
        g.despine(left=True)
        g.set_ylabels("client count")
        g.set_xlabels("AR")
    
    def app_decline_incr(self,binSize):
        df_r_decl = pd.DataFrame({'pr':self.pr_test,'y':self.y_test})
        df_r_decl['app'] = 'decline'
        df_r_app = pd.DataFrame({'pr':self.pr_train,'y':self.y_train})
        df_r_app['app'] = 'approve'
        df_r = pd.concat([df_r_decl,df_r_app]) 
        df_r = df_r.sort_values('pr').reset_index(drop=True)
        df_r['buck'] = 0
        sh = len(df_r)//binSize
        for i in range(binSize):
            df_r.loc[sh*i:sh*(i+1),'buck'] = i*(100//binSize)
        g = df_r.groupby('buck').mean()['y'].reset_index().rename(columns={'y':'total','buck':'AR'})
        app = df_r[df_r.app=='approve']
        g_app = app.groupby('buck').mean()['y'].reset_index().rename(columns={'y':'badRate','buck':'AR'})
        decl = df_r[df_r.app=='decline']
        g_decl = decl.groupby('buck').mean()['y'].reset_index().rename(columns={'y':'badRate','buck':'AR'})
        g['approve'] = g_app.badRate
        g['decline'] = g_decl.badRate
        g.index = g['AR']
        g = g.drop('AR',axis=1)
        plt.figure(); plt.plot(g)
        plt.legend(list(g.columns), loc='upper left')
    
def сheckIndex(s):
    if s.index.dtype_str=='category':
        return True
    else: return False

def warm2Columns(dfSrc,column1,column2,badFlag,binSize=10,countLimit=200):
    df = dfSrc[[column1,column2,badFlag]].copy()
    bins1 = np.unique(algos.quantile(df[column1], np.linspace(0, 1, binSize+1)))
    bins2 = np.unique(algos.quantile(df[column2], np.linspace(0, 1, binSize+1)))
    df[column1+'_bin'] = pd.tools.tile._bins_to_cuts(df[column1], bins1, include_lowest=True)
    df[column2+'_bin'] = pd.tools.tile._bins_to_cuts(df[column2], bins2, include_lowest=True)
    pvMean = df.pivot_table(badFlag,column1+'_bin',column2+'_bin',np.mean).fillna(0)
    pvSize = df.pivot_table(badFlag,column1+'_bin',column2+'_bin',np.size).fillna(0)
    
    if сheckIndex(pvSize):
        for ind in pvSize.index:
            for col in pvSize.columns:
                if np.isnan(pvSize.loc[ind,col].values[0][0]):
                    pvMean.loc[ind,col]=0
                elif pvSize.loc[ind,col].values[0][0]<countLimit:
                    pvMean.loc[ind,col]=0
    else:
        for ind in pvSize.index:
            for col in pvSize.columns:
                if np.isnan(pvSize.loc[ind,col]):
                    pvMean.loc[ind,col]=0
                elif pvSize.loc[ind,col] < countLimit:
                    pvMean.loc[ind,col]=0

    ss = sns.heatmap(pvMean,annot=True)
    return pvMean,pvSize

#функция rejectInference:
#decl - исходный набор данных (перед кластеризацией переменных), у которого есть непустые колонки clientKey и FirstDeclineRule
#goodColumns - список колонок, которые вошли в исходную модель. В формате ххх_cat_WOE
#clustInfo,woeInf - последние актуальные
#preLR - набор преобразованных в вое данных, на которых строилась модель. Включают в себя badFlag
#lr - исходная построенная модель

def rejectInference(decl,goodColumns,clustInfo,woeInfo,preLR,lr,buckets=100,badFlag='badMob3'):
    """функция rejectInference:
    decl - исходный набор данных (перед кластеризацией переменных), у которого есть непустые колонки clientKey и FirstDeclineRule
    goodColumns - список колонок, которые вошли в исходную модель. В формате ххх_cat_WOE
    clustInfo,woeInf - последние актуальные
    preLR - набор преобразованных в вое данных, на которых строилась модель. Включают в себя badFlag
    lr - исходная построенная модель"""
    if len(clustInfo)==0:
        clustInfo = pd.DataFrame(columns = ['categorical','minVal','maxVal','variable'])
    
    firstDeclineRule = pd.read_csv('C:/YandexDisk/Work/RevoBigScorring/FirstDeclineRule.csv',delimiter=';')
    rso = np.loadtxt('C:/YandexDisk/Work/RevoBigScorring/allMob3/rand.out',delimiter=',')
    X = preLR.drop(badFlag,axis=1).values
    
    goodColumnsDecl = [x for x in goodColumns if x not in ['badFpd','badMob3']]
    declPostWoe = ootTransform(decl,clustInfo,woeInfo,goodColumnsDecl)
    Xdecl = declPostWoe[list(preLR.drop('badMob3',axis=1).columns)].values
    pr_decl = lr.predict_proba(Xdecl)[:,1]
    declPostWoe['scoring'] = pr_decl
    declFirstDec = pd.concat([decl[['clientKey','FirstDeclineRule']],declPostWoe.scoring],axis=1)
    declFirstDec = pd.merge(declFirstDec,firstDeclineRule)
    
    brr = bucketRate(preLR[badFlag],lr.predict_proba(X)[:,1],buckets=buckets)
    brrCo = list(brr.CutOff)
    declFirstDec['AR_cum'] = (declFirstDec.scoring).apply(decAR,args=[brrCo])
    declFirstDec = pd.merge(declFirstDec,brr[['AR_cum','curBadRate']],how='left')
    declFirstDec['rand'] = rso[:len(declFirstDec)]*100
    declFirstDec['badRand'] = (declFirstDec.curBadRate*declFirstDec.badCoeff - declFirstDec.rand).apply(lambda x: 0.0 if x < 0 else 1.0)
    return declFirstDec[['clientKey','AR_cum','badRand']]

def generate_decl_from_src(src):
    cnxn = sql_connect()
    fdr = getFDR(cnxn=cnxn)
    client_date = getClientDate(cnxn=cnxn)

    dfVarDate = pd.merge(src,client_date)
    dfVarDate = pd.merge(dfVarDate,fdr)
    dfVarDate.applicationDate = pd.to_datetime(dfVarDate.applicationDate)
    lastDate = dfVarDate[dfVarDate['badMob3'].notnull()].sort_values('applicationDate',ascending=False)[1000:1001]['applicationDate'].values[0]
    dfDecl = dfVarDate[(dfVarDate['applicationDate']<=lastDate)&(dfVarDate.badMob3.isnull())]
    dfDecl.FirstDeclineRule = dfDecl.FirstDeclineRule.fillna('0')
    return dfDecl

def generate_decl_woe(src,ri,woe_info,good_columns,clust_info=[]): 
    """generating decl_woe from ri and src"""
    decl = pd.merge(ri,src)
    decl['badMob3'] = decl['badRand']
    decl_woe = ootTransform(decl,clust_info,woe_info,good_columns)
    return decl_woe

def generate_reject_badFlags_from_other_ri(decl,Allri,woCHri):
    allBad = pd.merge(pd.merge(decl[['clientKey','badMob3']],Allri[['clientKey','badRand']],how='left').rename(columns={'badRand':'badAll'}),
     woCHri[['clientKey','badRand']].rename(columns={'badRand':'badwoCH'}),how='left')
    hasBad = allBad[~allBad[['badMob3','badAll','badwoCH']].isnull().all(axis=1)]
    hasBad['bad'] = hasBad['badAll'].fillna(0) + hasBad['badwoCH'].fillna(0)
    hasBad = hasBad[['clientKey','bad']].rename(columns = {'bad':'badMob3'})
    return hasBad

def writeTransformInfo(folderName,clustInfo,woeInfo,goodColumns,rewrite=False):
    
    if len(clustInfo)==0:
        clustInfo = pd.DataFrame(columns = ['categorical','minVal','maxVal','variable'])
    
    folderNameFull = folderName + 'modelDescription'
    if not os.path.exists(folderNameFull):
        os.makedirs(folderNameFull)
    
    if os.path.isfile(folderNameFull + '/TransformInfo.xlsx') and not rewrite:
        print('File Have Already Existed')
        return
    
    writer = pd.ExcelWriter(folderNameFull + '/TransformInfo.xlsx')
    clustInfo.to_excel(writer,index=False,sheet_name='clustInfo')
    woeInfo.to_excel(writer,index=False,sheet_name='woeInfo')
    pd.DataFrame({'columns' : goodColumns}).to_excel(writer,index=False,sheet_name='goodColumns')    
    writer.save()

def write_src(folderName,src,rewrite=False):
    folderNameFull = folderName + 'modelDescription'
    if not os.path.exists(folderNameFull):
        os.makedirs(folderNameFull)
    
    if os.path.isfile(folderNameFull + '/src.csv') and not rewrite:
        print('File Have Already Existed')
        return
    
    src.to_csv(folderNameFull + '/src.csv', index=False)   

def write_reject_inference(folderName,ri,rewrite=False):
    folderNameFull = folderName + 'modelDescription'
    if not os.path.exists(folderNameFull):
        os.makedirs(folderNameFull)
    
    if os.path.isfile(folderNameFull + '/rejectInference.csv') and not rewrite:
        print('File Have Already Existed')
        return
    
    ri.to_csv(folderNameFull + '/rejectInference.csv', index=False) 
    
def read_model_data(folderName):
    transform_file_way = folderName + 'modelDescription/TransformInfo.xlsx'
    clust_info = pd.read_excel(transform_file_way,sheetname='clustInfo')
    woe_info = pd.read_excel(transform_file_way,sheetname='woeInfo')
    gcDf = pd.read_excel(transform_file_way,sheetname='goodColumns')
    good_columns = list(gcDf['columns'])
    
    src = pd.read_csv(folderName + 'modelDescription/src.csv')
    
    return clust_info, woe_info, good_columns, src

def read_woe_output(folderName):
    woe_out = pd.read_excel(folderName + 'modelDescription/woeOutput.xlsx',sheetname='woeOut')
    intercept = pd.read_excel(folderName + 'modelDescription/woeOutput.xlsx',sheetname='intercept').values[0][0]
    return woe_out, intercept

def modelOutput(folderName, woeInfo, goodColumns, model, gg,rewrite=False):
    lr = model.lr
    factor = 20 / log(2)
    offset = 600 - (factor*log(50))
    intercept = -(lr.intercept_/len(lr.coef_[0]))*factor + offset/len(lr.coef_[0])
    intercept = intercept[0]*len(lr.coef_[0])
    woeOut = woeOutput(woeInfo,goodColumns,model,factor)
    
    if not rewrite:
        return intercept, woeOut 
    
    woeProd = woeProduction(woeOut)
    
    folderNameFull = folderName
    if not os.path.exists(folderNameFull):
        os.makedirs(folderNameFull)
    
    if os.path.isfile(folderNameFull + '/woeOutput.xlsx') and not rewrite:
        print('File Have Already Existed')
        return intercept, woeOut
    
    writer = pd.ExcelWriter(folderNameFull + '/woeOutput.xlsx')
    woeOut.to_excel(writer,index=False,sheet_name='woeOut')
    woeProd.replace(100000000.00000,'inf').fillna('nan').to_excel(writer,index=False,sheet_name='woeProd')
    pd.DataFrame({'intercept':[intercept]}).to_excel(writer,index=False,sheet_name='intercept')
    gg.to_excel(writer,index=False, sheet_name = 'information_value')
    writer.save()
    
    return intercept, woeOut
    
def giniOnRejectInference(appDf,declDf,badFlag='badMob3'):
    declDf = declDf[list(appDf.columns)]
    appDeclDf = pd.concat([declDf,appDf])
    
    X_appDecl = appDeclDf.drop(badFlag,axis=1).values
    X_app = appDf.drop(badFlag,axis=1).values
    X_decl = declDf.drop(badFlag,axis=1).values
    y_appDecl = appDeclDf[badFlag].values
    y_app = appDf[badFlag].values
    y_decl = declDf[badFlag].values
    
    lr = LogisticRegression(C=0.1)
    lr.fit(X_appDecl,y_appDecl)
    pr_appDecl = lr.predict_proba(X_appDecl)[:,1]
    pr_app = lr.predict_proba(X_app)[:,1]
    pr_decl = lr.predict_proba(X_decl)[:,1]
    
    a,b,gini_appDecl = rocAuc(y_appDecl,pr_appDecl)
    a,b,gini_app = rocAuc(y_app,pr_app)
    a,b,gini_decl = rocAuc(y_decl,pr_decl)
    print("="*60)
    print('giniAll      = %.4f\nginiApproved = %.4f\nginiDeclined = %.4f' % (gini_appDecl,gini_app,gini_decl))
    print("="*60)
    return lr

def get_query(query,cnxn):
    try:
        query_result = pd.read_sql(query,cnxn)
    except:
        connectString = 'DRIVER={SQL Server};SERVER=dwh.int.revoplus.ru;DATABASE=Revo_DW;UID=e.migaev;PWD='
        pswd = getpass.getpass('DWH_Password: ')
        connectString = connectString + pswd
        cnxn = pyodbc.connect(connectString)
        query_result = pd.read_sql(query,cnxn)
    return query_result

def getFDR(cnxn=1):
    fdr = get_query(queryFDR,cnxn)
    return fdr

def getClientDate(cnxn=1):
    client_date = get_query(clientDateQuery,cnxn)
    return client_date

def desc_group(df,column):
    gr = df.groupby(column).size().reset_index().sort_values(0,ascending=False)
    return gr