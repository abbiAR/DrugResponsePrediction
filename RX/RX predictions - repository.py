import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix as cm
from scipy.stats import spearmanr

def get_vrx():
    vrx = pd.read_csv('RX dataset repository.csv')
    vrx.set_index('Unnamed: 0', inplace=True)
    vrx.drop(['Alpelisib', 'Nilotinib'], inplace=True) # 10 and 8 patients missing respectively
    del vrx['Pat_23'] # missing more than 5 drugs
    vrx[vrx<-100]=-100 # bring values to a lowest of -100
    vrx[vrx>200]=200 # bring values to a highest of 200 
    vrx=vrx.round(2)
    vrx=vrx[~vrx.index.duplicated(keep='first')]
    return vrx


def TML(srx, train_pat):
    src=srx[train_pat].copy() 

    order = src.isna().sum().sort_values(ascending=True) # sort values by number of missing vals
    good_order = order.loc[order<3].index #all patients with less than 3 missing values

    for col in order.index: #for each patient in the training data
        if src[col].isna().sum().sum() > 0: #if there are any missing values
            
            impute_idx = src.loc[src[col].isna()].index # values to be imputed
            train_idx = list(set(src.index)-set(impute_idx)) #values to be used for training a ML model for imputing values
            train_cols = good_order.tolist()
            if col in train_cols:
                train_cols.remove(col) # remove cell line that is to be imputed from training data 
            
            op=src.copy() #copy training data
            
            Xtr1= op.loc[train_idx][train_cols] #set training index and columns (drugs/patients)
            Xtr1.fillna(0, inplace=True) # fill missing values in training data with 0s
            Xts1= op.loc[impute_idx][train_cols] #set samples to be used for prediction.
            Xts1.fillna(0, inplace=True)
            
            ytr1= src.loc[train_idx][col].copy() #target values for training
            yts1= src.loc[impute_idx][col].copy()#true values 
            
            gbx1=GradientBoostingRegressor(max_depth = 1, n_estimators = 50)
            gbx1.fit(Xtr1,ytr1)
            
            newcol = gbx1.predict(Xts1)
            newcol = pd.Series(newcol, index=yts1.index)

            for idd in newcol.index:
                srx.loc[idd, col] = newcol.loc[idd] #fill in imputed values directly in srx
                
    srx=srx.round(2)
    return(srx)


def predict(vrx):
    i=0
    total={} # register results for each patient
    for patient in vrx.columns: # predict every patient seperately
        spear=[] # appending spearman results
        i+=1
        print(i)
        refined=[] #append final results to report for each patient
        
        for g in range(5): # n experiments to achieve mean results from
            trx=vrx[patient].copy() #current patient response values
            trx=trx.sample(frac=1) #random shuffle
            
            trx.dropna(how='any', inplace=True) #used for target values
            srx=vrx.loc[list(trx.index)].copy() #used for training

            results=[] # appending predicted values
            true=[] # appending true values

            train_pat=srx.columns.tolist()
            train_pat.remove(patient)
                        
            srx = TML(srx, train_pat) # impute missing values in training data

            for test_drug in trx.index: #for each drug measured for the patient cell line
                train_drug = srx.index.to_list()
                train_drug.remove(test_drug)

                Xtr=srx.loc[train_drug][train_pat].copy()
                Xts=srx.loc[test_drug][train_pat].copy()
                
                ytr=trx.loc[train_drug].copy()
                yts=trx.loc[test_drug].copy()
                            
                rf=RandomForestRegressor(n_estimators=100)
                rf.fit(Xtr,ytr)
                
                Xts=np.array(Xts).reshape(1,-1)
                Xts=pd.DataFrame(Xts, columns=Xtr.columns)
                yp=rf.predict(Xts)

                results.append(yp[0])
                true.append(yts)
                
            
            thr=30 #viability threshold for hits, <=30% viability indicates a hit
            res = pd.DataFrame([results,true], columns=list(trx.index)).T #predicted values
    
            ref=check_res(res, thr) #process outcome to get results for this patient
            refined.append(ref)#append the outcomes for this experiment (n=5)
            spear.append(spearmanr(true,results)[0]) # append spearman correlation for this experiment
            
        #get mean and standard deviation of the 5 experiments for this patient cell line

        refined_mean = [np.round(np.mean(x),3) for x in zip(*refined)]
        refined_std = [np.round(np.std(x),3) for x in zip(*refined)]

        spear_mean = np.round(np.mean(spear),3)
        spear_std = np.round(np.std(spear),3)

        #enter mean results, standard dev results as well as the mean spearmanr and pearson for the patient into dictionary
        total[patient]= refined_mean + refined_std + [spear_mean, spear_std] 
        
    return total #return all mean and stdev results for all patients


def check_res(res, thr):
    
    top_pred=res.sort_values(0, ascending=True).index[:5]
    actual_val = res.loc[top_pred][1]

    uthr=thr+10 # upper threshold for viability used for predictions

    res2=res[0].copy() 
    check2=res[1].copy()   

    res2[res2<=uthr]=1 #using upper threshold to identify a hit in predictions, since models prioritise precision over recal
    res2[res2>uthr]=0 

    check2[check2<=thr]=1 # using the actual threshold for true values
    check2[check2>thr]=0

    cc=cm(check2, res2)

    try:
        total_hits=cc[1][0]+cc[1][1]
        total_test=cc[0][1]+cc[1][1]
        total_caught=cc[1][1]
        all_drugs = len(check2)
        
    except:
        total_hits=np.nan
        total_test=np.nan
        total_caught=np.nan
        all_drugs = np.nan
        
    tot=[np.round(res[1].mean(),2), np.round(actual_val.mean(),2), total_hits, total_test, total_caught, all_drugs]
    return tot

if __name__ == '__main__':

    vrx= get_vrx() # get data
    total = predict(vrx) # generate models, predictions and results
    df=pd.DataFrame(total).T #convert results in dictionary into a dataframe and asign column labels below
    
    df.columns = ['all_drugs_mean_viability', 'top_5_pred_mean_viability', 'total_hits',
                   'drugs_tested_mean', 'drugs_caught_mean', 'total_n_drugs_available',
                   'all_drugs_mean_viability_stdev', 'top_5_pred_mean_viability_stdev',
                   'total_hits_stdev', 'drugs_tested_stdev',
                   'drugs_caught_stdev', 'total_n_drugs_available_stdev',
                   'spearmanR_all_drugs_mean', 'spearman_all_drugs_stdev']

    
    del df['total_n_drugs_available_stdev'] #not necessary for results
    del df['total_hits_stdev'] #not necessary for results
    

    

    
    
