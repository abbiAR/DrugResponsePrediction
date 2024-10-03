import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import spearmanr, pearsonr
import random

def get_data():

    df1=pd.read_csv('GDSC1_Log10_IC50.csv')
    df1.set_index('Unnamed: 0', inplace=True)
    
    df2 = pd.read_csv('GDSC1_TML_completed_matrix.csv')
    df2.set_index('Unnamed: 0', inplace=True)

    #targeted drugs considered. Exclusion criteria: pan-activity threshold of < 1uM 
    g1_1000_nm = ['azd2014', 'gw843682x', 'rtrail', 'torin 2', 'panobinostat', 'jw-7-52-1', 'ql-viii-58', 'doxorubicin', 'parp_0108', 'bi-2536', 'obatoclax mesylate', 'bortezomib', 'epothilone b', 'parp_9482', 'flavopiridol', 'azd7762', 'luminespib', 'wz3105', 'jnk-9l', 'sepantronium bromide', 'daporinad', 'dactolisib', 'fy026', 'tanespimycin', 'sn-38', 'belinostat', 'gemcitabine', 'paclitaxel', 'pd0325901', 'igfr_3801', 'cytarabine', 'cudc-101', 'lestaurtinib', 'omipalisib', 'trichostatin a', 'snx-2112', 'methotrexate', 'elesclomol', 'nsc319726', 'arry-520', 'tw 37', 'midostaurin', 'plk_6522', 'gsk1059615', 'cct245232', 'cct245467', 'dacinostat', 'vinblastine', 'trametinib', 'thz-2-102-1', 'docetaxel', 'ar-42', 'bryostatin 1', 'rapamycin', 'fy012', 'thapsigargin', 'cgp-60474', 'wye-125132', 'mitomycin-c', 'azd4877', 'zl109', 'vinorelbine', 'ispinesib mesylate', 'temsirolimus']

    #all drugs considered
    g1_0_nm=[]
    
    df1= df1.T.drop(g1_1000_nm) # remove non-targeted drugs, change to g1_0_nM to include all drugs
    df2drop = list(set(df2.columns)&set(g1_1000_nm))
    df2= df2.T.drop(df2drop)
    
    df1=df1.loc[df2.index]
    df1=df1[df2.columns]

    return df1, df2 


def drug_selection(dft, cor_n):
    df_c=df1.copy()
    vari = df_c.T.var().sort_values(ascending=False)
    half=int(len(vari)/2)
    topX = vari.iloc[:half].index # remove 50% of drugs based on lower variance
    df_c= df_c.loc[topX]
    corr=df_c.T.corr('spearman')
    drugs = list(corr.columns)
    random.Random(4232).shuffle(drugs)
    out=[]
    approved=[]
    for d in drugs:
        if d not in out:
            approved.append(d)
            line=corr[d]
            line=line.loc[line>cor_n] # correlation coefficient cutoff, drugs that are more than 0p2 pearson are thrown away
            out=out+list(line.index)   
        
    return(approved)


def predict(df1, df2):
    
    runseed1=random.randint(0,100000)#22938#85529#75027#9502#312351
    
    # SELECT A SUBSET OF UNKNOWN PATIENTS RANDOMLY
    test_patients = list(df1.T.sample(frac=0.2, random_state = 82634).index) #this seed guarantees that the patients have not been involved in TML matrix generation
    train_patients = list(df1.T.drop(test_patients).index)

    dft=df1.T.drop(test_patients)

    #split test set into test and validation (20% to 10% each)
    half=int(len(test_patients)/2)
    validation_patients = random.Random(7754).sample(test_patients, half)
    test_patients = list(set(test_patients)-set(validation_patients))
            
    #PREDICT FOR EACH CELL LINE IN TEST SET
    results={}
    steps = int(len(test_patients)/20)
    progress = np.arange(0,len(test_patients), steps)

    g1_drugs = drug_selection(dft, 0.42) #submit dataframe and correlation threshold to get a list of drugs for drug panel
    
    for patient in validation_patients[:10]:
        
        temp_spear=[]
        temp_pear=[]
        temp_rmse=[]
        temp_perc=[]
                
        try:
            for i in range(5):
                runseed2=random.randint(0,100000)#22938#85529#75027#9502#312351
    
                train_patients_idx=random.Random(runseed2).sample(train_patients, 100)
                
                test_drugs = list(set(df1.index)-set(g1_drugs)) 

                drug_set = g1_drugs
                    
                model = RandomForestRegressor(n_estimators=50,random_state=runseed1)

                yts=df1[patient].loc[test_drugs].copy()
                yts.dropna(how='any', inplace=True)
                
                ytr=df2[patient].loc[drug_set].copy()
                ytr.dropna(how='any', inplace=True)
                
                Xts=df2[train_patients_idx].loc[yts.index].copy()
                Xtr=df2[train_patients_idx].loc[ytr.index].copy()

                model.fit(Xtr,ytr)
                yp=model.predict(Xts)

                spear = spearmanr(yts, yp)
                pear=pearsonr(yts, yp)
                rmse = np.sqrt(mse(yts,yp))

                temp_spear.append(spear[0])
                temp_pear.append(pear[0])
                temp_rmse.append(rmse)

            
            for i in range(5):
                spear_mean = np.mean(temp_spear)
                spear_stdev = np.std(temp_spear)
                pear_mean = np.mean(temp_pear)
                pear_stdev = np.std(temp_pear)
                rmse_mean = np.mean(temp_rmse)
                rmse_stdev = np.std(temp_rmse)


            entry = [pear_mean, pear_stdev ,spear_mean, spear_stdev, rmse_mean, rmse_stdev]
            results[patient]=entry

        except Exception as e:
            print('fail: ', patient)
            print(e)

    return results

if __name__ == '__main__':
    df1, df2 = get_data()
    results = predict(df1, df2)
    dfr=pd.DataFrame(results).T
    labels = ['pear_mean', 'pear_stdev','spear_mean', 'spear_stdev','rmse_mean', 'rmse_stdev']
    dfr.columns=labels
    dfr=dfr.round(3)



    
