import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import spearmanr, pearsonr
import random

def get_data():

    df1=pd.read_csv('PRISM_primary_screen_processed.csv')
    df1.set_index('Unnamed: 0', inplace=True)    
    
    df2 = pd.read_csv('PRISM_TML_completed_matrix.csv')
    df2.set_index('Unnamed: 0', inplace=True)

    #selective drugs considered. Exclusion criteria: activity threshold of < 1uM in over 20% of cell lines
    g1_1000_nm = ['10-hydroxycamptothecin', '7-hydroxystaurosporine', '9-aminoacridine', 'a-674563', 'abt-751', 'amg900', 'ar-42', 'at-7519', 'at-9283', 'at13387', 'az20', 'azd2014', 'azd2858', 'azd3463', 'azd5438', 'azd7762', 'azd8055', 'azd8330', 'bay-11-7082', 'bay-11-7085', 'bgt226', 'bi-2536', 'bi-78d3', 'biib021', 'bix-01294', 'bms-265246', 'bms-387032', 'bnc105', 'bntx', 'cct129202', 'cct137690', 'cd-437', 'chir-124', 'cid-5458317', 'cr8-(r)', 'cudc-101', 'cudc-907', 'cyt-997', 'd-64131', 'er-27319', 'fk-866', 'g-1', 'gdc-0349', 'gdc-0980', 'gsk2126458', 'gsk461364', 'gw-843682x', 'ha14-1', 'hmn-214', 'idra-21', 'ikk-2-inhibitor-v', 'jk-184', 'jnj-26481585', 'jnj-7706621', 'jtc-801', 'k-858', 'kf-38789', 'kpt-185', 'kpt-276', 'kw-2478', 'kx2-391', 'ldn-212854', 'ly2183240', 'ly2603618', 'ly2606368', 'ly2608204', 'ly2801653', 'ly2874455', 'ly3023414', 'ly364947', 'm-344', 'mg-132', 'mk-1775', 'mln0128', 'nh125', 'nms-1286937', 'nms-e973', 'nsc-319726', 'nsc-3852', 'nsc-632839', 'nsc-663284', 'nsc-697923', 'nvp-auy922', 'nvp-bez235', 'nvp-tae226', 'nvp-tae684', 'onx-0914', 'osi-930', 'ots167', 'p276-00', 'pci-24781', 'pf-03758309', 'pf-04691502', 'pf-05212384', 'pf-477736', 'pfk-015', 'pha-665752', 'pha-793887', 'pha-848125', 'pi-103', 'pik-75', 'pki-179', 'pp-121', 'pp242', 'pu-h71', 'r306465', 'r547', 'rita', 'rta-408', 'sb-225002', 'sb-2343', 'sb-743921', 'sb-939', 'sc-144', 'sgi-1027', 'sib-1757', 'sn-38', 'sns-314', 'snx-2112', 'snx-5422', 'stf-118804', 'su3327', 'tak-733', 'tak-901', 'tas-103', 'tg-02', 'tg-101209', 'th-302', 'tms', 'tw-37', 'ver-49009', 'vlx600', 'vu0238429', 'wp1066', 'wp1130', 'wye-125132', 'wz-3146', 'xl888', 'y-320', 'ym-155', 'zm-306416', 'zm-447439', 'acivicin', 'adarotene', 'albendazole', 'alexidine', 'alvespimycin', 'alvocidib', 'amsacrine', 'anguidine', 'anisomycin', 'aphidicolin', 'aurora-a-inhibitor-i', 'azalomycin-b', 'barasertib-hqpa', 'bardoxolone', 'bardoxolone-methyl', 'belinostat', 'benzethonium', 'bortezomib', 'brefeldin-a', 'brilliant-green', 'bruceantin', 'buparlisib', 'buphenine', 'cabazitaxel', 'camptothecin', 'carfilzomib', 'carmofur', 'cephalomannine', 'ceritinib', 'cetrimonium', 'chloropyramine', 'ciclopirox', 'cladribine', 'clofarabine', 'cobimetinib', 'colchicine', 'combretastatin-a-4', 'copanlisib', 'crenolanib', 'crystal-violet', 'cyclocytidine', 'cycloheximide', 'dacinostat', 'danusertib', 'dasatinib', 'daunorubicin', 'delanzomib', 'dequalinium', 'deslanoside', 'dienogest', 'digitoxigenin', 'digitoxin', 'digoxigenin', 'digoxin', 'dinaciclib', 'diphenyleneiodonium', 'docetaxel', 'dolastatin-10', 'doxorubicin', 'ecamsule-triethanolamine', 'echinomycin', 'elesclomol', 'emetine', 'entinostat', 'epirubicin', 'epothilone-a', 'epothilone-b', 'epothilone-d', 'etomoxir', 'etoposide', 'evodiamine', 'exatecan-mesylate', 'fdcyd', 'fenbendazole', 'filanesib', 'floxuridine', 'flubendazole', 'fluvastatin', 'foretinib', 'fosbretabulin', 'gambogic-acid', 'ganetespib', 'gemcitabine', 'genz-644282', 'givinostat', 'gossypol', 'halofuginone', 'harringtonine', 'hesperadin', 'homoharringtonine', 'idarubicin', 'indibulin', 'irinotecan', 'ispinesib', 'ixabepilone', 'ixazomib', 'ixazomib-citrate', 'lanatoside-c', 'latanoprost', 'lestaurtinib', 'litronesib', 'maytansinol-isobutyrate', 'mebendazole', 'mechlorethamine', 'methiazole', 'methotrexate', 'midostaurin', 'mitomycin-c', 'mitoxantrone', 'mocetinostat', 'monensin', 'nanchangmycin', 'napabucasin', 'narasin', 'nemorubicin', 'nocodazole', 'obatoclax', 'octenidine', 'oprozomib', 'oridonin', 'ouabain', 'oxibendazole', 'oxiracetam', 'oxyphencyclimine', 'paclitaxel', 'panobinostat', 'parbendazole', 'peruvoside', 'pevonedistat', 'phenylmercuric-acetate', 'pitavastatin', 'plinabulin', 'podophyllotoxin', 'ponatinib', 'pralatrexate', 'proscillaridin-a', 'puromycin', 'pyrithione-zinc', 'resminostat', 'ribociclib', 'rigosertib', 'romidepsin', 'roquinimex', 'rubitecan', 'ryuvidine', 'salinomycin', 'sangivamycin', 'satraplatin', 'selinexor', 'serdemetan', 'sparfloxacin', 'stattic', 'talazoparib', 'taltobulin', 'tanespimycin', 'teniposide', 'tetramethylthiuram-monosulfide', 'tetrandrine', 'thiomersal', 'thiostrepton', 'thiram', 'tivantinib', 'topotecan', 'torin-1', 'torin-2', 'triapine', 'trichostatin-a', 'triptolide', 'uprosertib', 'valrubicin', 'vincristine', 'vinorelbine', 'volasertib', 'voreloxin', 'vorinostat', 'xanomeline']
    g1_0_nm=[]
    
    df1= df1.T.drop(g1_0_nm) # g1_1000_nm removes non-selective drugs, change to g1_0_nM to include all drugs
    df2drop = list(set(df2.columns)&set(g1_0_nm))
    df2= df2.T.drop(df2drop)

    #only remove when dealing with ALL compounds (these are controls)
    df1.drop(['mg-132','bortezomib'], inplace=True)
    df2.drop(['mg-132','bortezomib'], inplace=True) 
       
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
    print('generating correlation matrix for selection')
    for d in drugs:
        if d not in out:
            approved.append(d)
            line=corr[d]
            line=line.loc[line>cor_n] #remove drugs above specified correlation coefficient cutoff
            out=out+list(line.index)    
    
    return approved 


def predict(df1, df2):

    runseed1=random.randint(0,100000)#22938#85529#75027#9502#312351
    # SELECT A SUBSET OF UNKNOWN PATIENTS RANDOMLY
    test_patients = ['T98G_CENTRAL_NERVOUS_SYSTEM', 'J82_URINARY_TRACT', 'SNU1196_BILIARY_TRACT',
                     'MELJUSO_SKIN', 'NCIH727_LUNG', 'GMS10_CENTRAL_NERVOUS_SYSTEM', 'HT1080_SOFT_TISSUE',
                     'T47D_BREAST', 'SNU1214_UPPER_AERODIGESTIVE_TRACT', 'KYSE30_OESOPHAGUS', 'LCLC103H_LUNG',
                     'UACC257_SKIN', 'C32_SKIN', 'CL34_LARGE_INTESTINE', 'SNU201_CENTRAL_NERVOUS_SYSTEM',
                     'BCPAP_THYROID', 'SNU308_BILIARY_TRACT', 'RT112_URINARY_TRACT', 'TCCSUP_URINARY_TRACT',
                     'HCT15_LARGE_INTESTINE', 'RVH421_SKIN', 'NCIH1355_LUNG', 'SNU466_CENTRAL_NERVOUS_SYSTEM',
                     'HCC44_LUNG', 'NCIH1581_LUNG', 'A673_BONE', 'AN3CA_ENDOMETRIUM', 'HDQP1_BREAST',
                     'HEC6_ENDOMETRIUM', 'HEC251_ENDOMETRIUM', 'PANC0327_PANCREAS', 'SKNAS_AUTONOMIC_GANGLIA',
                     'NCIH226_LUNG', 'BECKER_CENTRAL_NERVOUS_SYSTEM', 'MSTO211H_PLEURA', 'LS411N_LARGE_INTESTINE',
                     'SF539_CENTRAL_NERVOUS_SYSTEM', 'CAPAN2_PANCREAS', 'LXF289_LUNG', 'JHUEM2_ENDOMETRIUM',
                     'MESSA_SOFT_TISSUE', 'KE39_STOMACH', 'SKOV3_OVARY', 'CAL27_UPPER_AERODIGESTIVE_TRACT',
                     'SKMES1_LUNG', 'RCC10RGB_KIDNEY', 'SKMEL24_SKIN', 'WM2664_SKIN', 'TTC709_SOFT_TISSUE',
                     'AGS_STOMACH', 'KMRC1_KIDNEY', 'GAMG_CENTRAL_NERVOUS_SYSTEM', 'HT29_LARGE_INTESTINE',
                     'BICR56_UPPER_AERODIGESTIVE_TRACT', 'NCIH2087_LUNG', 'SNU489_CENTRAL_NERVOUS_SYSTEM',
                     'KNS62_LUNG', 'TT2609C02_THYROID', 'HT1376_URINARY_TRACT', 'JHH1_LIVER', 'NCIH292_LUNG',
                     'NCIH838_LUNG', 'EWS502_BONE', 'SNU81_LARGE_INTESTINE', 'TEN_ENDOMETRIUM', 'RL952_ENDOMETRIUM',
                     'SNUC5_LARGE_INTESTINE', 'JHH4_LIVER', 'DETROIT562_UPPER_AERODIGESTIVE_TRACT', 'RCM1_LARGE_INTESTINE',
                     'IGROV1_OVARY', 'KYSE70_OESOPHAGUS', 'ECGI10_OESOPHAGUS', 'CCLFPEDS0001T_KIDNEY', 'HCC1395_BREAST', 'JHOS2_OVARY',
                     'HS729_SOFT_TISSUE', '647V_URINARY_TRACT', 'PANC0203_PANCREAS', 'KYSE510_OESOPHAGUS', 'NCIH28_PLEURA',
                     'TE11_OESOPHAGUS', 'MKN7_STOMACH', 'SNUC2A_LARGE_INTESTINE', 'JHH5_LIVER', '253JBV_URINARY_TRACT', 'HLF_LIVER',
                     'RMGI_OVARY', 'MIAPACA2_PANCREAS', 'ONCODG1_OVARY', 'EFE184_ENDOMETRIUM', 'NCIH1793_LUNG', 'OV7_OVARY',
                     'PK45H_PANCREAS', 'MDAMB435S_SKIN', 'IALM_LUNG', 'TM31_CENTRAL_NERVOUS_SYSTEM', 'BICR16_UPPER_AERODIGESTIVE_TRACT',
                     'MELHO_SKIN', 'JIMT1_BREAST', 'SNB75_CENTRAL_NERVOUS_SYSTEM', 'NCIH441_LUNG', 'CADOES1_BONE', 'SCABER_URINARY_TRACT']

    
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

    g1_drugs = drug_selection(dft, 0.0828)#submit dataframe and correlation threshold to get a list of drugs for drug panel
    input(len(g1_drugs))
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

                topX=45 #Check hit rates for top 10, 20, 30, 45 etc.
                predix = compare.sort_values(1).iloc[-topX:].index
                truix = compare.sort_values(0).iloc[-topX:].index
                overlap=len(set(predix)&set(truix))/topX #how many of the topX drugs are common between predicted and true
                temp_overlaps.append(overlap)

                hit_id = compare.sort_values(1).iloc[-topX:][0]
                hit_id[hit_id<1.7]=0 # threshold for hits = 1.7 and above
                hit_id[hit_id>=1.7]=1
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




    
