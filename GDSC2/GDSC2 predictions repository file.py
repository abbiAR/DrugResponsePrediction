import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import spearmanr, pearsonr
import random

def get_data():

    df1=pd.read_csv('GDSC2_Log10_IC50.csv')
    df1.set_index('Unnamed: 0', inplace=True)
    
    df2 = pd.read_csv('GDSC2_TML_completed_matrix.csv')
    df2.set_index('Unnamed: 0', inplace=True)

    #selective drugs considered. Exclusion criteria: activity threshold of < 1uM in over 20% of cell lines
    g1_1000_nm = ['sabutoclax', 'pevonedistat', 'vinorelbine', 'mitoxantrone', 'lmp744', 'dinaciclib', 'daporinad', 'azd7762', 'rapamycin', 'docetaxel', 'topotecan', 'podophyllotoxin bromide', 'gne-317', 'bi-2536', 'epirubicin', 'dasatinib', 'trametinib', 'dactinomycin', 'mk-1775', 'tanespimycin', 'azd8055', 'romidepsin', 'gemcitabine', 'methotrexate', 'luminespib', 'mg-132', 'cdk9_5576', 'dactolisib', 'obatoclax mesylate', 'telomerase inhibitor ix', 'bortezomib', 'cdk9_5038', 'pbd-288', 'sepantronium bromide', 'staurosporine', 'lestaurtinib', 'camptothecin', 'vinblastine', 'paclitaxel', 'teniposide', 'sn-38', 'eg5_9814']
    
    #all drugs considered
    g1_0_nm=[]
    
    df1= df1.T.drop(g1_1000_nm) # remove non-selective drugs, change to g1_0_nM to include all drugs
    df2drop = list(set(df2.columns)&set(g1_1000_nm))# remove non-selective drugs, change to g1_0_nM to include all drugs
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
            line=line.loc[line>cor_n] #remove drugs above specified correlation coefficient cutoff
            out=out+list(line.index)   
        
    return(approved)


def predict(df1, df2):
    
    runseed1=random.randint(0,100000)#22938#85529#75027#9502#312351
    
    # SELECT A SUBSET OF UNKNOWN PATIENTS RANDOMLY
    test_patients = list(df1.T.sample(frac=0.2, random_state = 82634).index) #this seed guarantees that the patients have not been involved in TML matrix generation, prevening information leakage
    train_patients = list(df1.T.drop(test_patients).index)

    dft=df1.T.drop(test_patients)

    #split test set into test and validation (from 20% of dataset to 10% each)
    half=int(len(test_patients)/2)
    validation_patients = random.Random(7754).sample(test_patients, half)
    test_patients = list(set(test_patients)-set(validation_patients))
            
    #PREDICT FOR EACH CELL LINE IN TEST SET
    results={}
    steps = int(len(test_patients)/20)
    progress = np.arange(0,len(test_patients), steps)

    g1_drugs = drug_selection(dft, 0.42) #submit dataframe and correlation threshold to get a list of drugs for drug panel
    
    for patient in validation_patients:
        
        temp_spear=[]
        temp_pear=[]
        temp_rmse=[]
        temp_perc=[]
        temp_hit_rates=[] 
        temp_overlaps=[] # overlap between predicted and true top X drugs.
                
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

                compare=pd.DataFrame([yts.values, yp], columns=Xts.index).T

                topX=10 #Check hit rates for top 10, 20, 30 etc.
                predix = compare.sort_values(1).iloc[-topX:].index
                truix = compare.sort_values(0).iloc[-topX:].index
                overlap=len(set(predix)&set(truix))/topX #how many of the topX drugs are common between predicted and true
                temp_overlaps.append(overlap)

                hit_id = compare.sort_values(1).iloc[-topX:][0]
                hit_id[hit_id<6]=0
                hit_id[hit_id>=6]=1
                hit_rate=hit_id.sum()/topX
                temp_hit_rates.append(hit_rate)
                
            
            spear_mean = np.mean(temp_spear)
            spear_stdev = np.std(temp_spear)
            pear_mean = np.mean(temp_pear)
            pear_stdev = np.std(temp_pear)
            rmse_mean = np.mean(temp_rmse)
            rmse_stdev = np.std(temp_rmse)           
            hit_rates_mean = np.mean(temp_hit_rates) 
            hit_rates_stdev = np.std(temp_hit_rates)
            overlaps_mean = np.mean(temp_overlaps) 
            overlaps_stdev = np.std(temp_overlaps)
        
            entry = [pear_mean, pear_stdev ,spear_mean, spear_stdev, rmse_mean, rmse_stdev,
                     hit_rates_mean, hit_rates_stdev, overlaps_mean, overlaps_stdev]

            results[patient]=entry

        except Exception as e:
            print('fail: ', patient)
            print(e)

    return results

if __name__ == '__main__':
    df1, df2 = get_data()
    results = predict(df1, df2)
    dfr=pd.DataFrame(results).T
    labels = ['pear_mean', 'pear_stdev','spear_mean', 'spear_stdev','rmse_mean', 'rmse_stdev',
              'hit_rates_mean', 'hit_rates_stdev', 'overlaps_mean', 'overlaps_stdev']
    dfr.columns=labels
    dfr=dfr.round(3)




    
