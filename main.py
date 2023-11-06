#kshitiz lakhotia project 3


import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import rfft
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy, iqr
from scipy.signal import periodogram as psd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from scipy import stats
import pickle


import warnings
warnings.filterwarnings("ignore")


insulin_data_df1=pd.read_csv('InsulinData.csv',low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)'])

cgm_data_df1=pd.read_csv('CGMData.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])



insulin_data_df1['date_time_stamp']=pd.to_datetime(insulin_data_df1['Date'] + ' ' + insulin_data_df1['Time'])
cgm_data_df1['date_time_stamp']=pd.to_datetime(cgm_data_df1['Date'] + ' ' + cgm_data_df1['Time'])




def processmealdata(insulin_data,cgm_data,dateid):
    insulin_df=insulin_data.copy()
    insulin_df=insulin_df.set_index('date_time_stamp')
    valid_timestamp_df=insulin_df.sort_values(by='date_time_stamp',ascending=True).dropna().reset_index()
    valid_timestamp_df['BWZ Carb Input (grams)'].replace(0.0,np.nan,inplace=True)
    valid_timestamp_df=valid_timestamp_df.dropna()
    valid_timestamp_df=valid_timestamp_df.reset_index().drop(columns='index')
    valid_timestamp_list, value, result, dates=[], 0, [],[]
    for ix,i in enumerate(valid_timestamp_df['date_time_stamp']):
        try:
            value=(valid_timestamp_df['date_time_stamp'][ix+1]-i).seconds / 60.0
            if value >= 120:
                valid_timestamp_list.append(i)
        except KeyError:
            break
    for idx,i in enumerate(valid_timestamp_list):
        start=pd.to_datetime(i - timedelta(minutes=30))
        end=pd.to_datetime(i + timedelta(minutes=120))
        date=i.date().strftime(dateid)
        meals = [i]
        meals.extend(cgm_data.loc[pd.to_datetime(cgm_data['Date'],format=dateid) == date].set_index('date_time_stamp').between_time(start_time=start.strftime('%H:%M:%S'),end_time=end.strftime('%H:%M:%S'))['Sensor Glucose (mg/dL)'].values.tolist())
        result.append(meals)
    result = pd.DataFrame(result)
    index=result.iloc[:,1:31].isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>6).dropna().index
    clean_data=result.drop(result.index[index]).reset_index().drop(columns='index')
    clean_data=pd.concat([clean_data.iloc[:,0], clean_data.iloc[:,1:31].interpolate(method='linear',axis=1)], axis =1)
    index_to_drop=clean_data.iloc[:,1:31].isna().sum(axis=1).replace(0,np.nan).dropna().index
    clean_data=clean_data.drop(result.index[index_to_drop]).reset_index().drop(columns='index')
    return clean_data



meal_data1=processmealdata(insulin_data_df1,cgm_data_df1,'%m/%d/%Y')
dates = meal_data1.iloc[:,0]
meal_data1.drop(meal_data1.columns[0], axis=1, inplace=True)





def buildmealfeaturematrix(clean_data):
    clean_data=clean_data.dropna().reset_index().drop(columns='index')
    first_max_power, second_max_power, third_max_power, fourth_max_power, fifth_max_power, sixth_max_power,=[],[],[],[],[],[]
    etr, IQR, psd1, psd2, psd3 = [],[], [], [], []
    for i in range(len(clean_data)):
        fft_array=abs((rfft(clean_data.iloc[:,0:30].iloc[i].values.tolist()))**2).tolist()
        sorted_fft=abs((rfft(clean_data.iloc[:,0:30].iloc[i].values.tolist()))**2).tolist()
        sorted_fft.sort()
        first_max_power.append(sorted_fft[-2])
        second_max_power.append(sorted_fft[-3])
        third_max_power.append(sorted_fft[-4])
        fourth_max_power.append(sorted_fft[-2])
        fifth_max_power.append(sorted_fft[-3])
        sixth_max_power.append(sorted_fft[-4])
        etr.append(entropy(clean_data.iloc[i]))
        IQR.append(iqr(clean_data.iloc[i]))

        _, p = psd(clean_data.iloc[i])
        psd1.append(np.mean(p[0:5]))
        psd2.append(np.mean(p[5:10]))
        psd3.append(np.mean(p[10:16]))

    feature_matrix=pd.DataFrame()
    feature_matrix['first_max_power']=first_max_power
    feature_matrix['second_max_power']=second_max_power
    feature_matrix['third_max_power']=third_max_power
    feature_matrix['fourth_max_power']=fourth_max_power
    feature_matrix['fifth_max_power']=fifth_max_power
    feature_matrix['sixth_max_power']=sixth_max_power
    tm=clean_data.iloc[:,22:25].idxmin(axis=1)
    maximum=clean_data.iloc[:,5:19].idxmax(axis=1)
    vel_max, vel_min, vel_mean, acc_min, acc_max, acc_mean=[],[],[],[],[],[]
    for i in range(len(clean_data)):
      first_diff = np.diff(clean_data.iloc[:,maximum[i]:tm[i]].iloc[i].tolist())
      second_diff = np.diff(np.diff(clean_data.iloc[:,maximum[i]:tm[i]].iloc[i].tolist()))
      vel_max.append(first_diff.max())
      vel_min.append(first_diff.min())
      vel_mean.append(np.mean(first_diff))
      acc_max.append(second_diff.max())
      acc_min.append(second_diff.min())
      acc_mean.append(np.mean(second_diff))
    feature_matrix['Velocity_maximum']=vel_max
    feature_matrix['Velocity_minimum']=vel_min
    feature_matrix['Velocity_mean']=vel_mean

    feature_matrix['Acceleration_maximum']=vel_max
    feature_matrix['Acceleration_minimum']=acc_min
    feature_matrix['Acceleration_mean']=acc_mean
    
    feature_matrix['Entropy'] = etr
    feature_matrix['IQR'] = IQR
    feature_matrix['PSD1'] = psd1
    feature_matrix['PSD2'] = psd2
    feature_matrix['PSD3'] = psd3
    return feature_matrix



meal_feature_matrix1=buildmealfeaturematrix(meal_data1)
normalized_df=(meal_feature_matrix1-meal_feature_matrix1.mean())/meal_feature_matrix1.std()
meal_feature_matrix1 = normalized_df



def processgroundtruth(insulin_data,valid_timestamp_list,dateid):
    insulin_df=insulin_data.copy()
    insulin_df=insulin_df.set_index('date_time_stamp')
    valid_timestamp_df=insulin_df.sort_values(by='date_time_stamp',ascending=True).dropna().reset_index()
    valid_timestamp_df['BWZ Carb Input (grams)'].replace(0.0,np.nan,inplace=True)
    valid_timestamp_df=valid_timestamp_df.dropna()
    valid_timestamp_df=valid_timestamp_df.reset_index()
    result=[]
    for idx,i in enumerate(valid_timestamp_list):
        date=i.date().strftime(dateid)
        result.append(valid_timestamp_df.loc[valid_timestamp_df['date_time_stamp']==i]['BWZ Carb Input (grams)'].values.tolist())
    return pd.DataFrame(result)



ground_truth = processgroundtruth(insulin_data_df1,dates, '%m/%d/%Y')



ground_truth



min = ground_truth[0].min()
max = ground_truth[0].max()
print(min, max)


import math
def extract_ground_truth(x2, min, max):
  result = []
  for i in range(len(x2[0])):
    result.append(math.floor((x2[0][i]-min)/20))
  return pd.DataFrame(result)


discrete_truth = extract_ground_truth(ground_truth, min, max)

print("Ground Truth Bins:","\n", discrete_truth.value_counts())


def Calc_Entropy(CalcValues):
    EntropyMealValue= []
    for InsulinValue in CalcValues:
      InsulinValue = np.array(InsulinValue)
      InsulinValue = InsulinValue / float(InsulinValue.sum())
      CalcValueEntropy = (InsulinValue * [ np.log2(glucose) if glucose!=0 else 0 for glucose in InsulinValue]).sum()
      EntropyMealValue += [CalcValueEntropy]
   
    return EntropyMealValue

def Purity(CalcValues):
    MealPurity = []
    for InsulinValue in CalcValues:
      InsulinValue = np.array(InsulinValue)
      InsulinValue = InsulinValue / float(InsulinValue.sum())
      CalcPurity = InsulinValue.max()
      MealPurity += [CalcPurity]
    return MealPurity



def CalcClusterMatrix(groundTruth,Clustered,k):
    Matrix= np.zeros((k, k))
    for i,j in enumerate(groundTruth):
         val1 = int(j)
         val2 = Clustered[i]
         if (val2 == -1):
           continue
         Matrix[val1,val2]+=1
    return Matrix


kmeans = KMeans(n_clusters=6,max_iter=7000)

pLabels=kmeans.fit_predict(meal_feature_matrix1)
df = pd.DataFrame()
df['kmeans_clusters']=pLabels 
print("Clusters Generated by K Means:","\n", df)


Matrix = CalcClusterMatrix(discrete_truth.iloc[:,0],pLabels,6)
MatrixEntropy = Calc_Entropy(Matrix)
MatrixPurity = Purity(Matrix)
Count = np.array([discrete_truth.sum() for discrete_truth in Matrix])
CountVal = Count / float(Count.sum())

KMeanSSE = kmeans.inertia_
KMeansPurity =  (MatrixPurity*CountVal).sum()
KMeansEntropy = -(MatrixEntropy*CountVal).sum()
print("Bin * Cluster Matrix for Kmeans", "\n", Matrix)



def CalcDBSCAN(dbscan,test,meal_pca2):
     for i in test.index:
         dbscan=0
         for index,row in meal_pca2[meal_pca2['clusters']==i].iterrows(): 
             test_row=list(test.iloc[0,:])
             meal_row=list(row[:-1])
             for j in range(0,12):
                 dbscan+=((test_row[j]-meal_row[j])**2)
     return dbscan



from collections import Counter
db = DBSCAN(eps=2.9, min_samples=4)
clusters = db.fit_predict(meal_feature_matrix1)
df_db = pd.DataFrame()
df_db['dbscan_clusters']=clusters 
print("Clusters Generated by DBSCAN:","\n", df_db)
count = Counter(clusters)
print(count.keys(), count.values())


DBSCANData=meal_feature_matrix1
DBSCANData['clusters']=list(clusters)
meal_feature_matrix1


initial_value=0
bins = 6
i = DBSCANData['clusters'].max()
while i<bins-1:
        MaxLabel=stats.mode(DBSCANData['clusters']).mode[0] 
        ClusterData=DBSCANData[DBSCANData['clusters']==MaxLabel]
        bi_kmeans= KMeans(n_clusters=2,max_iter=1000, algorithm = 'auto').fit(ClusterData)
        bi_pLabels=list(bi_kmeans.labels_)
        ClusterData['bi_pcluster']=bi_pLabels
        zeros=ClusterData[ClusterData['bi_pcluster']==0].index
        ones=ClusterData[ClusterData['bi_pcluster']==1].index
        max_cluster = DBSCANData['clusters'].max()
        for x in zeros:
            DBSCANData.loc[x,'clusters'] = MaxLabel
        for x in ones:
            DBSCANData.loc[x,'clusters'] = max_cluster+1
        i=i+1 

print("Clusters Generated by DBSCAN:","\n", DBSCANData.loc[:,'clusters'])
ct = Counter(DBSCANData.loc[:,'clusters'])
print(ct.keys(), ct.values())


MatrixDBSCAN = CalcClusterMatrix(discrete_truth.iloc[:,0],DBSCANData.loc[:,'clusters'],6)
    
ClusterEntropy = Calc_Entropy(MatrixDBSCAN)
ClusterPurity = Purity(MatrixDBSCAN)
Count = np.array([InsulinValue.sum() for InsulinValue in MatrixDBSCAN])
CountVal = Count / float(Count.sum())

Centroids = DBSCANData.groupby(DBSCANData['clusters']).mean()

dbscan = CalcDBSCAN(initial_value,Centroids,DBSCANData)
DBSCANPurity =  (ClusterPurity*CountVal).sum()        
DBSCANEntropy = -(ClusterEntropy*CountVal).sum()


print(MatrixDBSCAN)


print(dbscan, DBSCANPurity, DBSCANEntropy, KMeanSSE, KMeansPurity, KMeansEntropy)


FinalData = pd.DataFrame([[KMeanSSE,dbscan,KMeansEntropy,DBSCANEntropy,KMeansPurity,DBSCANPurity]],columns=['K-Means SSE','DBSCAN SSE','K-Means entropy','DBSCAN entropy','K-Means purity','DBSCAN purity'])
FinalData=FinalData.fillna(0)
FinalData.to_csv('Results.csv',index=False,header=None)

