```python
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
```


```python
#data from all train faults (dfta) and a fc table to translate the fault code acronyms in dfta.
dfta = pd.read_excel(r'C:\Darius\Darius\Downloads\Text Analysis\Fault.xls', parse_dates = ['Record Date'])
```


```python
#data cleaning
```


```python
#check number of columns in the dfta
dfta.columns.value_counts().sum()
```




    49




```python
#select only required columns for analysis
dfta1 = dfta[
    ['Record Date','Activity Type','Notification Number','FDS Remarks',
    'Fault Group']]
dfta1.columns = dfta1.columns.str.replace(" ","_")
dfta1.describe()

```

    C:\Users\Darius\AppData\Local\Temp/ipykernel_12848/2306398654.py:6: FutureWarning: Treating datetime data as categorical rather than numeric in `.describe` is deprecated and will be removed in a future version of pandas. Specify `datetime_is_numeric=True` to silence this warning and adopt the future behavior now.
      dfta1.describe()
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Record_Date</th>
      <th>Activity_Type</th>
      <th>Notification_Number</th>
      <th>FDS_Remarks</th>
      <th>Fault_Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>45094</td>
      <td>45092</td>
      <td>45094</td>
      <td>45094</td>
      <td>45094</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>38047</td>
      <td>11</td>
      <td>23343</td>
      <td>16659</td>
      <td>70</td>
    </tr>
    <tr>
      <th>top</th>
      <td>2013-03-30 23:44:57</td>
      <td>003</td>
      <td>Nil</td>
      <td>-</td>
      <td>Air Condition System</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>44</td>
      <td>41178</td>
      <td>21591</td>
      <td>27783</td>
      <td>6597</td>
    </tr>
    <tr>
      <th>first</th>
      <td>2010-01-01 06:37:35</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>last</th>
      <td>2021-12-01 20:16:49</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfta1['Activity_Type'] = dfta1['Activity_Type'].fillna('nil').copy()
```

    C:\Users\Darius\AppData\Local\Temp/ipykernel_12848/378331074.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      dfta1['Activity_Type'] = dfta1['Activity_Type'].fillna('nil').copy()
    


```python
#check how many FDS_remark contain only "-"
dfta1['FDS_Remarks'].str.contains(r"-$").value_counts()
```




    True     27790
    False    17304
    Name: FDS_Remarks, dtype: int64




```python
#clean the Notificaiton_Number column
dfta1 = dfta1[dfta1['Notification_Number'].notnull()].copy()
dfta1['Notification_Number'] = dfta1['Notification_Number'].str.replace("Nil","Non Service Request")
dfta1.head(5)
dfta1.shape
```




    (41178, 5)




```python
#check data type
dfta1.dtypes
```




    Record_Date            datetime64[ns]
    Activity_Type                  object
    Notification_Number            object
    FDS_Remarks                    object
    Fault_Group                    object
    dtype: object




```python
#check activitity unique values
dfta1["Activity_Type"].unique()
```




    array(['003'], dtype=object)




```python
#Select only specific Activity Type of maintneance work for this analysis
dfta1 = dfta1[dfta1["Activity_Type"].str.contains("003")]
dfta1filtered = dfta1[~dfta1['FDS_Remarks'].str.contains(r"-$")]
dfta1filtered.shape
```




    (16985, 5)




```python
#drop fault group with Nil comments
dfta1filtered = dfta1filtered[~dfta1['Fault_Group'].str.contains('Nil')].copy()
```

    C:\Users\Darius\AppData\Local\Temp/ipykernel_12848/2636388965.py:2: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      dfta1filtered = dfta1filtered[~dfta1['Fault_Group'].str.contains('Nil')].copy()
    


```python
dfta1filtered['Fault_Group'].unique()
```




    array(['ALS Propulsion System', 'Saloon Door System', 'Electrical System',
           'Saloon Air Conditioning System', 'Miscellaneous System',
           'Lighting Equipment', 'Pneumatic System', 'Bogie Equipment',
           'Brake System', 'ALS Auxiliary Power Supply Sys',
           'Emergency Door System', 'Miscellaneous', 'Axle/Wheel Set',
           'Air Condition System', 'Carbody Equipment', "Driver's Console",
           'Propulsion System', 'Train Wash Plant - Side Wash',
           'Train Wash Plant - Roof Wash', 'Coupler System',
           'Train Wash Plant', 'Plant Equipment  Train Wash Plant',
           'Fire Detection System', '*** Deleted on 25.08.2016 ***',
           'Lighting System', 'Train Integrated Management System (TIMS)',
           'Audio', 'Battery System', 'Pneumatic Supply System',
           'Auxiliary System', 'Interior Equipment', 'Train',
           '(DO NOT USE) Interior System', 'Gangway', 'PAPIS Equipment',
           'High Voltage', 'TCMS', 'Auxiliary Control Electronics',
           'COMMS - APU/ACU', 'Air Conditioning System'], dtype=object)




```python
#cleaning up the fault_Group
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains("TIMS|TCMS", regex = True)), "Fault_Group" ] = 'TIMS'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains("\*\*\*", regex = True)), "Fault_Group" ] = 'Others'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains('Saloon Air Conditioning System|Air Conditioning System|Air Condition System')), "Fault_Group"] = 'Others'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains('ALS Propulsion System|Propulsion System')), "Fault_Group"] = 'Propulsion System'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains('Emergency Door System|Saloon Door System')), "Fault_Group"] = 'Saloon Door System'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains('ALS Auxiliary Power Supply Sys|Auxiliary Control Electronics|Battery System|Auxiliary System')), "Fault_Group"] = 'Auxiliary System'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains('Pneumatic System|Pneumatic Supply System')), "Fault_Group"] = 'Pneumatic Supply System'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains('High Voltage|Electrical System')), "Fault_Group"] = 'Electrical System'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains('Train')), "Fault_Group"] = 'Others'     
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains("Coupler System|Bogie Equipment|Fire Detection System|Driver's Console|Axle/Wheel Set|Wheel & Axle|Wheel Slip / Slide System|Brake System|PAPIS Equipment|COMMS - APU/ACU")), "Fault_Group"] = 'Others_Critical'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains('Miscellaneous|Miscellaneous System|Brake Shoe|Plant Equipment  Underfloor Wheel Reprofiling System|Carbody Equipment|Interior Equipment|Lighting System|Audio|COMMS - CCTV/TTIS|Lighting Equipment|Gangway System|Gangway')), "Fault_Group"] = 'Others'      
```


```python
#differentiate critical and non critical category
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains("TIMS", regex = True)), "Criticality" ] = 'Critical Defects'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains("Others")), "Criticality"] = 'Non-Critical Defects'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains('Propulsion System')), "Criticality"] = 'Critical Defects'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains('Saloon Door System')), "Criticality"] = 'Critical Defects'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains('Auxiliary Control Electronics|Battery System|Auxiliary System')), "Criticality"] = 'Critical Defects'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains('Pneumatic Supply System')), "Criticality"] = 'Critical Defects'
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains('Electrical System')), "Criticality"] = 'Critical Defects'    
dfta1filtered.loc[(dfta1filtered["Fault_Group"].str.contains("Others_Critical")), "Criticality"] = 'Critical Defects'                   
```


```python
#check No. of rows in each fault group 
print(dfta1filtered['Fault_Group'].value_counts())
print(dfta1filtered['Fault_Group'].value_counts().sum())
```

    Others                          10961
    Others_Critical                  1669
    Propulsion System                1192
    Electrical System                 920
    Auxiliary System                  788
    TIMS                              695
    Pneumatic Supply System           378
    Saloon Door System                319
    (DO NOT USE) Interior System       33
    Name: Fault_Group, dtype: int64
    16955
    


```python
#(DO NOT USE) Interior System (old fault group) has a mixture of critical and non critical system. TO drop.
dfta1filtered = dfta1filtered.loc[~(dfta1filtered['Fault_Group'] == "(DO NOT USE) Interior System")]
dfta1filtered = dfta1filtered.loc[~(dfta1filtered['Fault_Group'] == "Nil")]
```


```python
#check No. of rows in each Criticality
print(dfta1filtered['Criticality'].value_counts())
print(dfta1filtered['Criticality'].value_counts().sum())
```

    Non-Critical Defects    10961
    Critical Defects         5961
    Name: Criticality, dtype: int64
    16922
    


```python
print(dfta1filtered['Fault_Group'].value_counts())
print(dfta1filtered['Fault_Group'].value_counts().sum())
```

    Others                     10961
    Others_Critical             1669
    Propulsion System           1192
    Electrical System            920
    Auxiliary System             788
    TIMS                         695
    Pneumatic Supply System      378
    Saloon Door System           319
    Name: Fault_Group, dtype: int64
    16922
    


```python
dfta1filtered.reset_index(drop=True, inplace=True)
```


```python
dfta1filtered.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Record_Date</th>
      <th>Activity_Type</th>
      <th>Notification_Number</th>
      <th>FDS_Remarks</th>
      <th>Fault_Group</th>
      <th>Criticality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-08-01 08:00:49</td>
      <td>003</td>
      <td>FDRSC0267499</td>
      <td>TIP under brake system shown  Car 2 BCE2 'mino...</td>
      <td>Propulsion System</td>
      <td>Critical Defects</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010-12-01 06:05:00</td>
      <td>003</td>
      <td>FDRSC0267914</td>
      <td>TTIS did not display doors closing whentrain d...</td>
      <td>Saloon Door System</td>
      <td>Critical Defects</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010-12-01 08:28:27</td>
      <td>003</td>
      <td>FDRSC0267933</td>
      <td>TIP and ATS Alarm Manager showed Door and EHS ...</td>
      <td>Electrical System</td>
      <td>Critical Defects</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010-12-01 17:33:05</td>
      <td>003</td>
      <td>FDRSC0267973</td>
      <td>TIP showed that car 2 cooling failed,rover (CS...</td>
      <td>Others</td>
      <td>Non-Critical Defects</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010-01-13 06:06:52</td>
      <td>003</td>
      <td>FDRSC0268015</td>
      <td>CSO Rover reported that TTIS for Cars showing ...</td>
      <td>Electrical System</td>
      <td>Critical Defects</td>
    </tr>
  </tbody>
</table>
</div>




```python
features = dfta1filtered.loc[:,["FDS_Remarks","Criticality"]]
```


```python
#Prepare the data for machine learning modeling

#1 convert Fault Group to a numerical variable
features['Criticality'] = features['Criticality'].map({'Critical Defects':1, 'Non-Critical Defects':0}).copy()
features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FDS_Remarks</th>
      <th>Criticality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TIP under brake system shown  Car 2 BCE2 'mino...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TTIS did not display doors closing whentrain d...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TIP and ATS Alarm Manager showed Door and EHS ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TIP showed that car 2 cooling failed,rover (CS...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CSO Rover reported that TTIS for Cars showing ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#2 Define X and y (from the FDS data) for use with COUNTVECTORIZER
X = features.FDS_Remarks
y = features.Criticality
print(X.shape)
print(y.shape)
```

    (16922,)
    (16922,)
    


```python
#3 split X and y into training and testing sets
# by default, it splits 75% training and 25% test
# random_state=1 for reproducibility
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

    (12691,)
    (4231,)
    (12691,)
    (4231,)
    


```python
#4 import and instantiate CountVectorizer (with the default parameters)
from sklearn.feature_extraction.text import CountVectorizer

#5 instantiate the vectorizer
vect = CountVectorizer()
```


```python
#6 combine fit and transform 
X_train_ft = vect.fit_transform(X_train)
X_train_ft
```




    <12691x7194 sparse matrix of type '<class 'numpy.int64'>'
    	with 274193 stored elements in Compressed Sparse Row format>




```python
#7 transform testing data, using fitted vocabulary from X_train, into a document-term matrix. 4239 rows.
X_test_ft = vect.transform(X_test)
X_test_ft
```




    <4231x7194 sparse matrix of type '<class 'numpy.int64'>'
    	with 89830 stored elements in Compressed Sparse Row format>




```python
#train the model for machine learning
```


```python
#1 import
from sklearn.naive_bayes import MultinomialNB

#2 instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()
```


```python
#3 train the model 

nb.fit(X_train_ft, y_train)
```




    MultinomialNB()




```python
#4 make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_ft)
```


```python
#5 calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)
```




    0.8267549042779485




```python
y_test.value_counts().sum()
```




    4231




```python
# examine class distribution
print(y_test.value_counts())
# there is a majority class of 0 here, which is expected

# calculate null accuracy. lower than accuracy of class prediction which is okay.
null_accuracy = y_test.value_counts().head(1) / len(y_test)
print('Null accuracy:' , null_accuracy)

```

    0    2732
    1    1499
    Name: Criticality, dtype: int64
    Null accuracy: 0    0.64571
    Name: Criticality, dtype: float64
    


```python
# print the confusion matrix using heatmap to present
sns.heatmap(metrics.confusion_matrix(y_test, y_pred_class),annot=True, fmt="d")
```




    <AxesSubplot:>




    
![png](output_35_1.png)
    



```python
# print message text for the false positives for non critical cases (or false negative for critical cases) which is more severe than the counter part.

list(X_test[(y_pred_class==0) & (y_test==1)])
```




    ['ATS Alarm Manager & TIP showed Train PV22 Car 3: Interior smoke detection - Fault detected. Asset: 8223/RSC/FSD. Informed RS Najib.',
     'ATS Alarm Manager and TIP showed Train PV29 Car 2: Interior Smoke Detection - Pre Alarm Detected. Informed RS Siva. Asset: 8292/RSC/FSD',
     'ATS alarm manager indicated PV20/svc43 underseat fire detected at Car3. Rover on board reported no signs of smoke or fire.',
     'ATS alarm manager and TIP showed car 3 active cab DD cover open, HLV staff was activated to normalised the alarm, but unable, staff alighted at KRG. RSMCO imformed.',
     'ATS Alarm Manager & TIP showed Train PV36 Car 2: Interior Smoke Detection - Fault Detected. InformedRS Najib. Asset:8362/RSC/FSD',
     'ATS Alarm Manager and TIP showed Train PV25 Car 3: Interior smoke detection - Fault detected.Informed RS Catherine..\tAsset: 8253/RSC/FSD',
     'ISCS Alarm Manager & TIP showed Train PV05 Car 1: Interior Smoke Detection - Fault Detected. Informed RSMCO Siva. Asset: 8051/RSC/FSD',
     'PV06 Car1 A1 door leaving a small gap when closing.',
     'ATS Alarm Manager & TIP showed Train PV24 Car 1: Air cooling function - Failed. Informed RS Siva. Asset: 8241/RSC/ACN\t',
     'ATS Alarm Manager and TIP showed Train PV17 Car 3: Detrainment door - DD cover open. Asset: 8173/RSC/XDOR. Informed RS Tan  Y T.',
     'ATS Alarm Manager and TIP showed Train PV25 Car 2: Under Frame Fire Detection - Failure.Informed RSTony.\tAsset : 8252/RSC/FSD',
     'ATS Alarm Manager and TIP showed Train PV29 Car 1: Interior Smoke Detection . Fault Detected.Informed RS Tan Y T.\tAsset: 8291/RSC/FSD',
     'PV62/65: ATS alarms & TIP showed Car2 & Car3 Interior Smoke Detection Fault.',
     'TIP showed DD cover open at CAR 1, ATO all failed inttermitten and EHS showing unknown at Car 2 andCar 3. Rover on board check DDU, all is normalised and DD cover is closed on site but TIP and ATS alarm still showed the',
     'ATS Alarm Manager & TIP showed PV35 Car2: Interior Smoke Detection - Fault Detected. Informed RS  Tony. \tAsset: 8352/RSC/FSD',
     'ATS Alarm Manager and TIP showed Train PV06  Car 1: Under Seat Fire Detection- Under Seat Detected.Informed  RS Siva. Asset: 8061/RSC/FSD\t',
     'ATS Alarm Manager showed Train PV07 Car2: Interior Smoke Detection - Pre-Alarm Detected. Informed RS Najib. Asset: 8072/RSC/FSD\t',
     'Passenger reported at HBF that there isa gap at one of the train doors. SS boarded at LBD, verifiedthere is a small gap about 2 cm at Car3 A4 door. SS pushedthe door closed tightly manually.',
     'ATS Alarm Manager and TIP showed Train PV05 Car 1: Interior Smoke Detection - Fault Detected. Informed RS Jack. Asset: 8051/RSC/FSD\t',
     'ATS Alarm Manager & TIP showed Train PV07 Car 1 & Car 3 : Air Cooling Function - Failed. Informed RS\tYT Tan. Asset: 8071/RSC/CAN; 8073/RSC/CAN',
     'PV07 EB by ATP with ITAMA removed. Rover on board reported train overrun by 2 PSDs, beyond setback limit. PV07 proceeded in CM to BLY without pax exchange at SER OT. Pax was detrained at BLY OT and proceeded back to depot. Trip cancellation from BLY OT to PYL OT. ATS alarm indicated Fatal fault AM, CM and RM not available on standby ATP.',
     'ATS Alarm Manager & TIP showed PV21 Car3: Interior Smoke Detection - Fault Detected. Asset:  8213/RSC/ FSD. Informed RS Tony.',
     'CSO Rover reported that TTIS for Cars showing " <Station_name> for all stations. Informed RSMCO andComms.',
     'Train PV38 Rover reported that Car1 Aircon was warmed and no ventilation. ATS Alarm Manager and TIP showed Air Cooling Failed and Ventilation Major. Informed DSM Jack. Stockchanged with PV33 at DBG.',
     'Rover onboard feedback that Car 1 A4 door is opening at 4secs later than the rest of the train doors. Rover checked and found no obstructions to the door.   RSMCO YT informed.',
     'ATS Alarm Manager and TIP showed Train PV18 Car1: Interior smoke detection - Fault detected. Informed RS\tJack. Asset : 8181/RSC/FSD.',
     'ATS Alarm Manager and TIP showed Train PV25 Car 3: Interior smoke detection - Fault detected. Informed RS Tony. Asset: 8253/RSC/FSD\t',
     'Car3 EHS cover detected opened. Rover reported all seal not broken but car3 b1 cover is loose.',
     'Intermittent Car1 B2 door and Car2 B2 door push back alarm. DSM informed. Rover reported train doorsokay.',
     'OCC unable to select Trainborne CCTV .Informed Comms   who quickly did a status check when train moving from MRM IT to LRC OT. Lawrence  reported the CCTV function was working fine, suspected the CCTV image not sent to O',
     'ATS Alarm Manager and TIP showed Train PV07 Car 1: Air cooling function Failed. Asset: 8071/RSC/ACN.\tInformed RS Jack.',
     'Station staff informed that theres a paxcomplained that  PV30 T-Car A1 door was closing so hard andyou can hear a very loud sound. MPS staff board the train and found out that the door rubber came out. Station staff ma',
     'ATS Alarm Manager & TIP showed Train PV12 Car 3: Interior Smoke Detection - Fault Detected. InformedRSMCO Tony. Asset: 8123/RSC/FSD',
     'ATS Alarm Manager & TIP showed Train PV01 Car 3: Interior Smoke Detection - Fault Detected. InformedRS Najib. Asset: 8013/RSC/FSD.\t',
     'ATS Alarm Manager showed toggling alarmsof Train PV17: Air Compressor Failed at Car 3 and Forced atCar 1. It also showed that Air Cooling Function failed forALL Cars. Informed RS Najib. Asset: 8173/RSC/PNEU/COMP\t',
     'PSDs recycled to open position due to obstacle detection alarm at Car 3, B3 doors. RSM Norizan.',
     'ATS Alarm Manager & TIP showed Train PV30 Car 1: Interior smoke detection - Fault detected. InformedRS Tony. Asset: 8301/RSC/FSD',
     'PV41/41: Car2 alarms and TIP showed Interior Smoke Detection - Pre Alarm. Rover confirmed no heat or smoke.',
     'BSH SS reported that pax feedback airconnot cold. KRG SS checked and feedback okay. DSM informed.',
     'ATS Alarm Manager and TIP showed Train PV03 Car3: Interior smoke detection - Fault detected.InformedRS Tan Y T. Asset : 8033/RSC/FSD.',
     'Rover reported train doors remained openwhen PSDs closed. TCO1 sent ID to train a few times for train doors & PSDs to close togeather. SIG Ng KK.',
     'ATS Alarm Manager & TIP showed Train PV06 Car 3: Interior smoke detection - Fault detected.InformedRS Najib. Asset: 8063/RSC/FSD',
     'PV49: TIP showed ACU/APU at Car1 and 3 failed. PEI and PA not operational. Rover on board reported no PA in train. Train Stockchange at SDM with PV 64. DSM informed',
     'Intermittent Car1 B2 door and Car2 B2 door push back alarm. DSM informed.',
     'PV48: DRMD Car3 A1 door not working. DSM and COMMS informed.',
     'ATS Alarm Manager & TIP showed Train  PV40 : Undeframe Fire Detection- Failure at Car 2\t. Informed RS Tan Y T. Asset: 8402/RSC/FSD',
     'ATS Alarm Manager and TIP showed PV21: Car 1: Under Seat Fire Detection Failure. Informed RS YT. \tAsset: 8211/RSC/FFSD\t',
     'ATS Alarm Manager & TIP showed PV21 Car3: Interior Smoke Detection - Fault Detected. Asset:  8213/RSC/FSD. Informed RS Siva.',
     'Rover on board informed that at T-car, neutral isolation 2 is in failure. RS aware of fault.',
     'Car1 underseat fire detection. DKT SS activated to check and reported no abnormality. RSMCO informed.',
     'PV10: Rover reported that aircon condensation leaking near Car3 1st pair of doors. Stock changed with PV14 at PYL.',
     'ATS Alarm Manager & TIP showed Train PV30 Car 1: Air cooling function - Failed.Informed RS Tan Y T.\tAsset:8301/RSC/ACN',
     'Rover reported car3 b1 train door indicator light not working.',
     'Unable to depart LRC IT due to train doors at Car1 B2 door not proven closed and locked. LRC SS force close doors and train departed. Train info page indicate door is closed but not locked. At CDT IT; rover on board isol',
     'Observation from CCTV; Train doors and PSDs opened normally for pax exchange. ATS alarm panel indicated Car 3 B3 Door push back alarm 8 to 9 seconds later. Fault normalised after 8 to 9 seconds. (TSG to PYL OT)  Alarm ap',
     'Event viewer showed toggling alarm for Train PV19: Emergency handle switch saloon door - Unknown.\tAsset: 8191/RSC/SDOR/EHS/A1 to A4; 8192/RSC/SDOR/EHS/A1 to A4;8193/RSC/SDOR/EHS/A1 to A4; 8193/RSC/SDOR/EHS/B1 to B4. Inf',
     'Train PV55 Car3 Interior Smoke Detection - Pre Alarmed Detected showed in the ATS Alarm Manager and TIP. CDT Rover confirmed no sign of smoke and heat. Informed DSM Jack',
     'Rover reported that aircon condensationleaking at Car1 near driving console. Train was sent back toKCD after revenue service. Replaced with PV10. DSM informed',
     'ATS alarm and TIP showed PV 07 ESH coverdetected opened at Car 2 B2 door. Rover checked ESH cover was loose and unable to normalise the EHS cover. RSMCO was informed.',
     'ATS Alarm Manager and TIP showed Train PV26 Car 3: Interior Smoke Detection - Fault Detected.Informed RS Tony. Asset: 8263/RSC/FSD\t',
     'ATS alarm manager and TIP showed car 1 seat fire detected.  HPV SS was activated to check and reported no sign of fire or smoke.',
     'ATS alarm manager and TIP showed Train PV23 Car 1: Detrainment door - DD cover open. Asset: 8231/RSC/XDOR. Informed RS Y T Tan.',
     'Rover reported PV50 Car 3 DRMD near A1 & A2 doors not working.',
     'TIP showed PV26/6 Car 2 had intermittentalarms of Exterior Smoke Detection, Interior Smoke Detection - Fault Detected, Under Frame/Seat Fire Detetcion - Failure, Air Cooling Function - Unknown, Ventilation Function - Un',
     'ATS Alarm Manager & TIP showed PV15 Car2 & 3: Interior Smoke Detection - Fault Detected. Informed RS Najib. Asset: 8152/RSC/FSD;  8153/RSC/FSD\t',
     'Train unable to depart DKT OT. Train doors and PSD kept opening and closing. Instructed CSO to switch to CM and close train and PSD doors to continue svc. Trainwas not able to proceed due to CM/AM not available. Pax wer',
     'ATS alarm manager and TIP showed Car 2 and 3  EHS; A and B side train doors toggling fault to unknown state.    RS Yap informed.',
     'ATS alarm manager and TIP showed car3 seat fire detected. ONH SS checked and reported no sign of smoke and fire.',
     'ATS Alarm Manager and TIP showed Train PV31CAR2 :Door - Door Failure Closed. Informed RS Najib. Asset:8312/RSC/SDOR/SDOR/B2',
     'ATS Alarm Manager and TIP showed Train PV15 Car 2: Interior Smoke Detection - Fault Detected. InformedRS Jack. Asset: 8152/RSC/FSD\t',
     'ATS Alarm Manager and TIP showed Train PV25 Car 2: Under Frame Fire Detection - Failure. Informed RSMCO Najib.Asset : 8252/RSC/FSD',
     'CSO ( rover ) reported console cover atcar1 unable to open. Discovered by RSM Halim that the yale key lock set was loose. RSMCO Tony informed. Stockchange wasarranged during peak hour withdrawal.',
     'PV12 ATS alarm manager and TIP showed car 1 normal lighting status failed. Rover feedback car 1 half lighting only. Train was stock changed.',
     'SNAP REP 8153: Reported glass panel vibrating quite strong on carriage 8153 near A2 door when train moving. PV15 stable at S9D. DSM informed.',
     'Rover reported of heavy condensation atCar1 and Car2 gangway. Stockchanged at PYL replace by PV17',
     'ATS Alarm Manager & TIP showed Train PV32 Car 3: Air Cooling Function - Failed. Informed RSMCO Siva.Asset: 8323/RSC/ACN',
     'ATS alarm manager and TIP showed EB by Obstacle Dectection and self normalised. DSM informed.',
     'PV60/Svc27 Car 3 Interior Smoke Detection-Fault Detected.\nPV64/Svc49 Car 3 Interior Smoke Detection-Fault Detected.',
     'PV41: Rover reported that train door and PSDs door recycled at BSH OT. ATS alarm showed ATC; Car2 saloon doors not closed and locked. TCO2 send ID for train to depart in AM.\nCCTV playback showed train doors recycled once while all PSDS remain close. \nDev Dep at BSH OT: 00: 41sec.\n\nTSC Syafiq -5',
     'ATS Alarm Manager & TIP showed Train PV32 Car 2: Interior Smoke Detection - Fault Detected. InformedRS Tony. Asset :8322/RSC/FSD\t',
     'ATS Alarm Manager & TIP showed Train PV07 Car 1: Air Cooling Function - Failed. Informed RS Najib. Asset: 8071/RSC/ACN\t',
     'ATS Alarm Manager and TIP showed Train PV25 Car 3: Interior smoke detection - Fault detected. Informed RS Jack. Asset: 8253/RSC/FSD\t',
     'PV64/60: Alarms & TIP displayed Interior Smoke Alarm - Pre Alarm detected. Rover check and confirmed no heat / smoke.',
     'ATS alarm manager and TIP showed car1; car2 and car3 door failure closed. Train was manned.',
     'PV61: SCO unable to make live train PA. Intrain PA is normal. Comms and DSM informed.',
     'Abnormal noise reported at trailer car A2 door when closing. PV doors able to close and lock normally.',
     'ATS Alarm Manager & TIP showed Train PV38 Car 3: Interior smoke detection - Fault detected. \tAsset:8383/RSC/FSD. Informed RS Tony.',
     'ATS Alarm Manager & TIP showed Train PV21 Car 1: Air cooling function - Failed. Informed  RS Tony. Asset: 8211/RSC/ACN.\t',
     'ATS Alarm Manager & TIP showed Train PV01 Car 3: Interior Smoke Detection - Fault detected. Asset: 8013/RSC/FSD. Informed RS Y T Tan.',
     'TIP showed: Smoke; Pre Alarm at Car2. BLY SS checked - no heat/ smoke. RSMCO informed.',
     'Pv01 Car 1 B1 door when door closing very noisy.',
     'ATS Alarm Manager & TIP showed Train PV31 Car 2 Under frame Fire detection : Interior smoke detection; Under seat fire dection: Informed RSMCO YT Tan.   Asset :8312/RSC/ACN',
     'ATS Alarm Manager and TIP showed Train PV18 Car1: Interior smoke detection - Fault detected. Informed RS Tan. Asset : 8181/RSC/FSD.',
     'PV60 ATS alarm showing Car1 interior smoke Detection fault detected alarm.',
     'ATS Alarm Manager & TIP showed PV22 Car2: Interior Smoke Detection - Fault Detected. Informed RS Najib.\tAsset: 8222/RSC/FSD',
     'PV55: Unable to establish in-train PA. Error Notification showed Train Agent Protocol Failure.',
     'ATS Alarm Manager & TIP showed Train PV32 Car 2 & 3: Interior Smoke Detection - Fault Detected. Informed RS Najib. Asset :8322/RSC/FSD\t  ; 8323/RSC/FSD',
     'Rover reported that the EHS Cover for Car1 B2 is loose but the seal is still intact. RSMCO Yap informed.',
     'ATS Alarm Manager showed Train PV26: Unknown Status. Informed RSMCO TanYT. Asset: 8262/RSC/ATC/xxx',
     'Rover reported car3 DD cover no seal. Informed RSMCO.',
     'ATS Alarm Manager & TIP showed Train PV14 Car 3: Interior Smoke Detection - Fault Detected. InformedRSMCO Tony. Asset: 8143/RSC/FSD',
     'ATS Alarm Manager showed numerous toggling alarms of EHS Saloon unknown for A1,A2.A3 and A4 door  for Car 3 and Door unknown status. Informed RS Tony.\tAsset:8383/RSXC/SDOR/EHS',
     'ATS alarm manager and TIP showed Train PV37 Car 3: Detrainment door - DD cover open. Informed RS Nagib. Asset: 8373/RSC/XDOR\t',
     'ATS Alarm Manager & TIP showed Train PV31 Car 2: Under Frame Fire Detection -  Under Seat Fire Detection  and Interior Smoke Detection - Unknown status. Informed RS Najib.\t Asset: 8312/RSC/FSD',
     'SNAP REP reported that 8131 has an annoying high pitch sound while door are open. Rover on board instructed to check all doors and feedback no such fault encountered. RS informed.',
     'PV45 Zabbix command 73 alarm. Unable to make PA - error train agent protocol. PV to run in mainline with console cover open to watch out for PEC alarm. RS & Comms informed. At 0733hrs PV stock change with PV09 PYL IT /MT.',
     'ATS Alarm Manager & TIP showed PV34 Car1: Detrainment Door - DD Cover Open. Asset: 8341/RSC/XDOR. Informed RS Y T Tan.',
     'CSO rover reported DDU at CAR 1 Touch screen faulty.',
     'TIP showed Car1 and Car2 Air Cooling Function Failed. BSH SS checked and confirmed Car1 and Car2 warm. Stockchanged with PV10 at MRB. DSM informed.',
     'PV55 PEC activated but no response from pax. TSG SS checked and confirmed that no assistance required. After PEC was remotely reset by OCC, PV55 was still unable to depart the Platform due DIR remained Open, with all train door closed & locked. SS was asked to detrain the passengers and Rover onboard closed the Traiin Doors in CM and activated the DIRBS. Train was withdrawn in CM form TSG IT to TSG RT3. \nRSM Jason.',
     'AT BLY IT Car2 A2 door not proven closed, door did not close fully. Rover have to close the door fully and isolated the door. Immediate departure sent. PSD of faulty door opened at SER IT and LRC IT.',
     'ATS Alarm Manager and TIP showed toggling alarms of Train PV40: Emergency Handle Switch Saloon Door& Door - Unknown. Scheduled withdrawn. Asset: 8401/RSC/SDOR/EHS/A1 ; A2; A3; A4; B2; B3;  8401/RSC/SDOR/SDOR/A1; A2; A3;',
     'ATS Alarm Manager & TIP showed Train PV31 Car 2: Under Frame Fire Detection -  Under Seat Fire Detection  and Interior Smoke Detection - Unknown status. Informed RS Siva. Asset: 8312/RSC/FSD\t',
     'ATS alarm manager and TIP showed car 1 smoke fault.',
     'Reported by rover - Car 1 B4 door EHS unit loose.',
     'ATS Alarm Manager and TIP showed Train PV29 Car 1: Interior Smoke Detection . Fault Detected.  Informed RS Tony. Asset: 8291/RSC/FSD',
     'ATS Alarm Manager and TIP showed Train PV21 Car2: Interior smoke detection - Fault detected. Asset :8213/RSC/FSD. Informed RS Najib.',
     'ATS Alarm Manager & TIP showed Train PV38 Car 3: Front Display Unit Status - Unknown: Blue. InformedRSMCO Tony. Asset: 8383/RSC/CAB/FDU',
     'PV30, reported via SNAPREP, Car 1 Air Con Warm. With reference to RIMS 54567, PV30 already stock changed at 0910hrs at PYL.',
     'PV12 CSO Rover reported that for the DDUshown Car 1Cooling Failed and Neutral Isolation 1 for T Carfailed. Actual temperature for Car 1is ok. TIP also shown failed for Car 1. Informed RSMCO Najeeb',
     'ATS alarm manager and TIP showed push back alarm at B2 side door for Car 1; Car 2 and Car 3.',
     'ATS Alarm Manager and TIP showed Train PV05 Car 1: Interior smoke detection - Fault detected. Asset: 8051/RSC/FSD. Informed RS Jack.',
     'ATS Alarm Manager & TIP showed Train PV31 Car 2: Under Frame Fire Detection - Unknown; Under Seat Fire Detection - Unknown & Interior Smoke Detection - Unknown.Informed RS Siva.  Asset: 8312/RSC/FSD',
     'ATS Alarm Manager & TIP showed PV22 Car3: Interior Smoke Detection - Fault Detected. Informed RS Najib.\tAsset: 8223/RSC/FSD\t',
     'ATS alarm manager showed toggling alarmsfor PV23:Air Cooling Function Unknown,Air Conditiong FreshAir Dampers - Fails To Closed at Car3 A1.Likewise for the TIP.These alarms were generated and self-normalised after a fe',
     'Rover reported Car3 DDU having blank screen.  RSM checked and reported loose wire for limit switch.Adjusted and ok.',
     'Rover reported car 3 A1 door slow in opening. RSMCO informed.',
     'Smoke dectection  alarm',
     'ATS Alarm Manager and TIP showed Train PV34 Car 13 Interior smoke detection - Fault detected. Asset:8343/RSC/FSD. Informed RS Nagib.',
     'PV25 Car1 door A1 didnot open at every station.',
     'Rover reported car2 neutral isolation 2fail.',
     'At KRG IT PV23 unable to depart due to door obstruction at Car 1 A3. SS clear obstruction and trainwas wable to depart. RSM Alvin',
     'Rover reported hissing sound was heard at Car 3 driving console location. DSM informed.',
     'ATS Alarm Manager & TIP showed Train PV22 Car 3: Interior smoke detection - Fault detected. Asset: 8223/RSC/FSD. Informed RS Siva.',
     'RSM reported that at Car3 near A1 door of PV07, there was squeaking sound when train was departing from station.',
     'Rover reported car 3 DDU display blank.',
     'ATS Alarm Manager & TIP showed Train PV05 Car 1: Interior Smoke Detection - Fault Detected. InformedRSMCO Siva. Asset: 8051/RSC/FSD',
     'Rover reported that the lock mechanismfor the detrainment door  cover for car 3 is loose. Rover confirmed that cover is closed and locked but TIP shows Coveris opened. RSMCO Yap informed.',
     'Exterior Smoke Detected for Car 1 showedin the ATS Alarm Manager and TIP. Rover confirmed no sign of heat or smoke. Informed RSMCO Jack.',
     'ATS Alarm Manager & TIP showed Train PV30 Car 1: Interior smoke detection - Fault detected.InformedRS\tYT Tan. Asset: 8301/RSC/FSD\t',
     'ATS Alarm Manager & TIP showed Train PV33 Car 1: Interior smoke detection - Fault detected. Asset: 8331/RSC/FSD. Informed RS YT Tan.',
     'At SDM OT train overrun less then 0.5 min AM mode. Train auto set back and normal train doors opened and closed.   There was no in train PA announcement for next station and close door at all stations. TTIS display was',
     'Rover reported that Car 1 A1 door slowerin opening as compared to the rest.  RS Yap informed.',
     'PV 46 Car 3 A2 DRMD',
     'ATS Alarm Manager indicated Car3 B2 PushBack  Intermittetnt alarm.RSM was informed.',
     'Rover on board reported that Train PV10 underrun half PSD at Platform. Train jogged forward 3times to precise stop for passengers exchanged. Rover monitored 3 stations ahead and found no abnormalities. Informed DSM Tony and Signal OCC Senthil.',
     'Door Failure(close position) at Car 1 B2door. Rover tripped DMC4CB but fault unable to clear.',
     'ATS Alarm Manager and TIP showed Train PV24 Car 1: Air cooling function - Failed.Informed RS Y T Tan. \tAsset: 8241/RSC/ACN',
     'ATS Alarm Manager and TIP showed Train PV23 Car1: Interior smoke detection - Fault detected. Asset :8231/RSC/FSD. Informed RS Jack',
     'Rover reported the sticker at the detrainment door car 3  "Fine $5000 for miss use" was torn.',
     'ATS Alarm Manager showed alarms for Train PV39: Air Conditioning Return Air Dampers - Fails to open(Car 1 - A1, A2, B1 & B2, Car 2 - A2 & B2, Car 3 - B1 & B2).Informed RS Siva. Asset: 8391/RSC/CAN/RDM/A1, A2, B1 & B2 a',
     'EPN SS reported of vibrations from PV42/svc30, Car3 near A2 door from ceiling area. Confirmed by NCH SS. Stockchanged arranged.',
     'ATS alarm indicated PV49/svc38 ACU/APU failed at Car3. Rover reported in train PA is normal.',
     'ATS Alarm Manager and TIP showed Train PV25 Car 2: Under Frame Fire Detection - Failure. Informed RSY T Tan. Asset : 8252/RSC/FSD.\t',
     'Passengers feedback to SS that Car1 A1 door has a gap of 2-3 cm when fully closed. Activated SS to verify, no obstruction or loose contact at the door. RSMCO (Jack) Informed',
     'DD cover alarm at car 3 unable to reset.',
     'ATS Alarm Manager showed PV10 having Ventilation Major Fault at Car 2. TIP showed the same. InformedRSMCO Siva. Asset: 8102/RSC/ACN',
     'PV38 at S9E unable to wake up. Checked info page found ATO and RIOM failed. Informed RS Siva.',
     'ATS Alarm Manager & TIP showed Train PV32 Car 2: Interior smoke detection - Fault detected. Asset :8322/RSC/FSD. Informed RS Catherine.',
     'ATS alarm manager showed interval of 1 min toggling alarm for PV08:Air Conditioning fresh Air Dampers - Fails To Close at Car3 A1 B1,Wheel Slip/Slide - At Least1 Slipping or One Sliding Detected.Likewise for the TIP.The',
     'Train PV17 Car 3 Under Seat Fire Detection shown in the ATS alarm manager and TIP. BLY ASM board the train and confirmed that there is no sign of smoke and heat. Informed DSM Jack.',
     'Car2 B1 door failure open showed at ATSalarm. SS activated to checked and reported door did not fully closed. SS assisted to push closed train door. Train doordid not fully closed again at DKT OT. PYL SS assisted to cl',
     'ATS alarm manager and TIP showed Train PV22 Car 1: Detrainment Door cover open. Informed RS Yap. Asset: 8221/RSC/XDOR\t',
     'Rover reported that the Y-disc at the Car 1 detrainment door dropped off and kept in the B-side detrainment cabinet.  RSMCO Yap informed.',
     'PV01 at BSH IT, all PSD and train doorsclosed and locked except for Car1 A3 door did not closed. BSH staff went onboard PV01 to ioslate train door.  Informed RSMCO',
     'ATS Alarm Manager & TIP showed Train PV05 Car 1: Interior smoke detection - Fault detected. Asset: 8051/RSC/FSD. Informed RS Siva.',
     'ATS Alarm Manager & TIP showed Train PV22 Car 3: Interior Smoke Detection - Fault Detected. InformedRS Siva. Asset: 8223/RSC/FSD\t',
     'Rover reported Car 3 Seven Seaters GlassPanel loose near A1 Door. Informed DSM Jack. Stockchanged with PV30 at PYL.',
     'ATS Alarm Manager & TIP showed Train PV39 Car 2: Air Cooling Function - Failed. Informed RSMCO Nagib. Asset: 8392/RSC/ACN',
     'Rover reported car2 neutral isolation 1failed. RSMCO informed.',
     'ATS alarm manager and TIP showed EB by Obstacle Dectection and self normalised. DSM informed.',
     'ATS Alarm Manager and TIP showed Train PV19 Car1: Interior smoke detection - Fault detected. Informed RS Siva.  Asset: 8191/RSC/FSD.',
     'ATS Alarm Manager and TIP showed Train PV38 Car 2: Interior smoke detection - Fault detected. Informed RS Jack. Asset: 8382/RSC/FSD',
     'ATS Alarm Manager and TIP showed Emergency Handle Cover opened at Car3 on PV07. Stn staff reported Car3 A2 door EHS cover was loose. After the cover was pushed,EHS cover was proven closed and locked.',
     'ATS Alarm Manager & TIP showed Train PV31 Car 2: Under Seat Fire Detection & Interior Smoke Detection - Unknown. Informed RSMCO Siva. Asset: 8312/RSC/RSD',
     'Rover onboard PV57/SVC22 reported TTIS at Car3 and Car2 is not working.',
     'ATS alarm manager and TIP showed Train PV13 Car 1: Detrainment door - DD cover open. Informed RS Yap. Asset: 8131/RSC/XDOR\t',
     'PV61/40: Zibbix Cmd73 train having communication issue alarm. OCC unable to broadcast in-train PA to train. Rover opened console cover to monitor for PEC activation. Fault self-recovered at HBF OT.',
     'ATS Alarm Manager and TIP showed Train PV25 Car 3: Interior smoke detection - Fault detected. Informed RS YT. Asset: 8253/RSC/FSD.\t',
     'ATS Alarm Manager and TIP showed Unknownstatus of Train PV31 Car 2: Under Frame Fire Detection; Under Seat Fire Detection & Interior Smoke Detection. InformedRS Tony.\t Asset: 8312/RSC/FSD',
     'SNAPREP PV18: Pax feedback 8181 strong burning smell.\nRover confirmed it was braking smell, stock change arranged at PYL.',
     'Train door at Car 1 B4 was closing veryslowly and rover had to push door manually to fully close and lock.',
     'ATS Alarm Manager & TIP showed Train PV24 Car 2: Interior Smoke Detection - Fault Detected. \tInformed RS Najib. Asset:8242/RSC/FSD',
     'Station Staff feedback that Car 2 B1 door is not completely closed. There is a gap of 5mm to 10mm between the doors.  RSMCO Najip informed.',
     'MC2 EHS cover panel of B3 EHS is loose.Screw not tight. Reported by Rover who checked. Stockchangewith PV16/Svc 40 at PYL.',
     'TIP and ATS Alarm Manager showed Car1 Air comp failed toggling. Informed RS Tony.',
     'ATS Alarm Manager and TIP showed Train PV07 Car 1: Air cooling function Failed. Asset: 8071/RSC/ACN.\tInformed RS YT.',
     'PV59, ATS alarm indicated Car3 ACU/ APU (Audio Amplifier ) - all failed. Rover feedback PA ok. DSM was informed.',
     'ATS Alarm Manager & TIP showed PV02 Car3: Interior Smoke Detection - Fault Detected. Informed RS Najib.\tAsset: 8023/RSC/FSD\t',
     'Alarm manager / TIP showed PV14 Car1 Under Seat / Underframe Fire Detected; Interior Smoke Detection. SER SS checked and confirmed no smoke or fire detected. DSM confirmed this was a known fault.',
     'Rover on board Pv48 from TSG to PYL reported the DRMD at Car 3 A1 door is not working.',
     'ATS alarm indicated Exterior Smoke Detected for Car 1. SS instructed to verify and found no sign ofheat and smoke. DSM YT informed.',
     'PV34 at LBD_MT unable to receive wake-upcommand. PPJ station staff proceed to PV34 and reported that train was in shut down mode. SS wake up train manually.',
     'ATS Alarm Manager & TIP showed Train PV01 Car 2: Interior Smoke Detection - Fault Detected. InformedRSMCO Jack. Asset: 8012/RSC/FSD',
     'At HBF OT, rover reported Car3 A1 door opened slower than the other train doors. Fault did not happen at other stations. RSMCO YT informed.',
     'PV59 Car2 , smoke Pre Alarm  on the TIP. CDT Rover SS confirmed no abnormalities.',
     'PV64 ATS alarm showed Car3 Interior Smoke detection - pre alarm detected. Rover onboard reported no heat and smoke .  DSM informed.',
     'ATS Alarm Manager and TIP showed Train PV26 Car 3: Interior Smoke Detection - Fault Detected. Informed RS Najib. Asset: 8263/RSC/FSD\t',
     'PV 34 Car 1 B1 door unable to be detected closed and locked. Rover tried to lock from inside and outside train but still failed. Train was sent to DBG IT Overrun for fault to be rectified later.  Asset: 8341/RSC/REL/DCLR',
     'ATS Alarm Manager and TIP showed Train PV37 Car 1: Interior Smoke Detection - Fault Detected. Informed RS Najib. Asset: 8371/RSC/FSD\t',
     'Rover reported Car3 driving console keyslot was difficult to open.',
     'PV45: PEC B4 and B2 at Car 1 activated at HBF S1. Rover on board checked, no pax was on board. PEC can be reset. SIG and RS informed.',
     'ATS Alarm Manager and TIP showed toggling Train PV14: Air Cooling Function - Failed at all cars. Informed RS JacK. Asset: 814X/RSC/ACN\t',
     'Toggling ATS Alarm recieved- PV26/05 hadCompressor FAILED. Informed RS MCO NAJIB',
     'ATS Alarm Manager and TIP showed PV21: Car 1: Under Seat Fire Detection Failure. Informed RS Jack.\tAsset: 8211/RSC/FSD.\t',
     'Train did not depart on schedule after pax exchange with train and PSDs remained opened.  No ATS andISCS alarms received.  TSC issued ID.  Delayed departure.RSM Sin Chee Soon',
     'ATS Alarm Manager & TIP showed Train PV32 Car 3: Air cooling function - Failed. Asset: 8323/RSC/ACN.\tInformed RS Najib.',
     'ATS Alarm Manager and TIP showed PV31 Car 2- Underframe Fire Detection ; Under seat fire detection;Interior Smoke Detection in unknown status,\tAsset:  8312/RSC/FSD. Informed RS Najib.',
     'No overspeed alarm at Car3 by Rover onboard. Informed DSM Tony',
     'PV07 not able to depart in AM due to alltrain doors did not close, hence, all PSD recycled at BBS IT. SS on board confirmed train doors were not closed. Faultoccured at every STN. STN staff closed doors in CM and proce',
     'Car2 cooling failed and Ventilation in Major fault. Rover to trip air con breaker to clear fault. DSM was informed.',
     'ATS Alarm Manager and TIP showed Train PV32 Car 3: Interior Smoke Detection - Fault Detected.Informed RS Najib. Asset: 8323/RSC/FSD',
     'CSO(Rover) reported PV34 Car1 Y-tool drop off. Rover placed it in the cabinet where mode selector was placed. RSMCO YT informed.',
     'ATS alarm manager andTIP showed PV30 MPUfaiiled>=1. Informed RS Najib',
     'ATS Alarm Manager showed Train PV19 Detrainment Door DD Cover Open Informed RSMCO Asset 8191/RSC/XDOR',
     'Car 2, A2 train door open slowly at eachstation. RS Najeeb informed',
     'ATS alarm page indicate Air Compressor 1, Car 1, Car 2, Car 3 cooling failed and normalised.',
     'Car 1 Ext. Smoke Failure.',
     'Rover onboard reported that alternate train doors did not open for service at PMN IT in CM mode.  All PSDs opened.  Active Cab 1 with B1 & B3 train doors did not open.  RS Yap informed.',
     'ATS Alarm Manager & TIP showed Train PV14 Car 1: Interior smoke detection - Fault detected. Asset: 8141/RSC/FSD. Informed RS Siva.',
     'PV02/76: ATS alarms & TIP displayed Car3: PEC & EHS for A-side doors; Air-con Cooling / Ventilation in Unknown. Informed DSM. Stock change at PYL 1918hrs.',
     'ATS Alarm Manager and TIP showed Train PV10 Car 1: Interior Smoke Detection - Fault Detected.Informed RS Tony.Asset: 8101/RSC/FSD',
     'Rover reported Half Ligthing for all cars. Air Cooling Function Failed for all cars showed in the ATS Alarm Manager and TIP. Stockchanged with PV31.',
     'PV50 Live PA to the train is loud. Feedback by pax and confirmed by rover.  \nPV47 , PV56  Live PA loud feedback by Rover\nDSM and Comms informed.  Live PA on safe Distancing and Good Morning PA.',
     'Rover reported car 3 detrainment door cover security seal was broken.',
     'ATS Alarm Manager and TIP showed Train PV32 Car 2: Interior Smoke Detection - Fault Detected. Informed RS\tSiva. Asset: 8322/RSC/FSD\t',
     'ATS Alarm Manager and TIP showed Train PV16 Car2: Interior smoke detection - Fault detected. Informed RS \tSiva. Asset: 8162/RSC/FSD',
     'Rover reported there is a small gap at Car 3 B1 door upon door full closure. Activated HLV SS to verify, SS reported there is no more gap but detected air flowing in due to rubber worn out. RSMCO (Jack) informed.',
     'Train PV59 PECU activated  for Car 1 Door A3. Before TCO acknowledge the call on the Alarm  banner the PECU normalised by itself. CDBG SS boarded  the Train confirmed no PEC lighted up on the PEC and DDU showed no activation. Informed DSM Tony. Stockchanged with PV47 at PYL.',
     'All PSDs closed but not Train Doors. PSDs reycled 3 times but not train doors. Finally the Train Doors closed with the PSDs and departed when immediate departure was sent by TCO. SS monitored PSDs were ok. Informed DSM S',
     'ATS Alarm Manager and TIP showed Train PV39 Car 3: Interior Smoke Detection - Fault Detected.Informed RS Tony.Asset: 8393/RSC/FSD',
     'ATS alarm manager and TIP showed car 1 smoke fault.',
     'PV25 at S11A, TIP showed Car2 Door C&L Relay Open alarm.and TTIS Car2 VMC Failure. Train was asleepand wake up a few times yet fault did not clear.',
     'ATS Alarm Manager & TIP showed Train PV13 Car 2: Interior Smoke Detection - Fault Detected. InformedRSMCO TanYT. Asset: 8132/RSC/FSD',
     'Smoke detection alarm in car 2.',
     'ATS Alarm Manager and TIP showed Train PV30 Car 2: Interior smoke detection - Fault detected. Informed RS Jack. Asset: 8302/RSC/FSD\t',
     'ATS Alarm Manager & TIP showed Train PV05 Car 1: Interior Smoke Detection - Fault Detected. InformedRSMCO TanYT. Asset: 8051/RSC/FSD',
     'ATS alarm manager and TIP showed car 2 and car 3 smoke fault.',
     'ATS Alarm Manager & TIP showed Train PV10 Car 1: Interior smoke detection - Fault detected. Asset: 8101/RSC/FSD. Informed RS Najib.',
     'PV34 TIP shown EHS Activation for A1 toA4 CAR 1 unknown. PEC A1 and A3 unknown. Extinguisher and Ext Smoke unknown status. Door A1 to A4  Car 1Command and CrewSwitch  Status unknown. Aircon & Ventilation System Car 1st',
     'ATS Alarm Manager & TIP showed Train PV40 Car 2: Under Frame Fire Detection - Failure. Informed RS Nagib. Asset: 8402/RSC/FSD\t',
     'Train PV05 PECU Test for Car1 Door A3 Comms NOK. CCTV and Alarm was OK. Informed DSM Tony. Try other PECU Car1 Door B2 was OK.',
     'CSO Rover reported that the A2 door Car1 have a rubbing sound when its closed. TIP status shown okfor this door. Inform RSMCO Najeeb.',
     'ATS Alarm Manager & TIP showed Train PV36 Car 1: Interior Smoke Detection - Fault Detected. InformedRSMCO Siva. Asset: 8361/RSC/FSD',
     'PV12 T-Car A1 door vibrating abnormallywhen train is moving. Rover verified but informed door opening/closing is normal and train still safe for pax servce. RSMCO informed.',
     'ATS alarm manager and TIP showed Train PV13 Car 3: Emergency handle switch saloon door - Unknown forall doors. Asset: 8133/RSC/SDOR/EHS. Informed RS Nagib',
     'ATS Alarm Manager and TIP showed Train PV06  Car 1: Under Seat Fire Deection- Under Seat Detected. Informed RS Jack. Asset: 8061/RSC/FSD',
     'ATS Alarm Manager and TIP showed Train PV32 Car 3: Interior Smoke Detection - Fault Detected. Informed RS\tYT Tan. Asset: 8323/RSC/FSD\t',
     'PV42. RSM feedback that PV42 car1 undercarriage near first door found sound like chopping board. Audio sound was send to DSM. Schedule withdrawal. DSM informed.',
     'ATS Alarm Manager and TIP showed Train PV09 Car2: Interior smoke detection - Fault detected. Asset :8092/RSC/FSD. Informed RS Siva.',
     'ATS Alarm Manager & TIP showed togglingof  Train PV32 Car3: Air Cooling Function - Failed. InformedRS Tony. Asset: 8323/RSC/ACN',
     'Rover reported car2 neutral isolation 2failed.',
     'Rover reported of heavy condensation between Car1 and Car2 (gangway). Stockchanged at PYL replaced by PV24',
     'ATS Alarm Manager and TIP showed Train PV19 Car1: Interior smoke detection - Fault detected. Informed RS\tTan Y  T.Asset: 8191/RSC/FSD.\t',
     'SNAP REP: PV22 Car1 air-con is poor. Train in depot. DSM informed.',
     'PV27: CM driving refresher. CCTV showed train door opened but PSDs not opened. TCO take back in AM mode. All doors opened for pax exchange.ERM/PSD06 was activated. PSD 06 was pushed close for train to depart by SS.TCO Patrick.',
     'ATS Alarm Manager and TIP showed Train PV03 Car3: Interior smoke detection - Fault detected. Asset :8033/RSC/FSD. Informed RS Najib.',
     'ATS Alarm Manager and TIP showed Train PV31 Car 2: Under Frame Fire Detection; Interior Smoke Detection & Underseat Fire Detection in Unknown status. Informed RS Jack. Asset: 8312/RSC/FSD',
     'ATS Alarm Manager & TIP showed PV40 Car3: Interior smoke detection - Fault detected. Asset: 8403/RSC/FSD. Informed RS Siva.',
     'PEC A3 & A2 Car 2 activated by pax whoseone child had rushed ahead into the train / stroller caughtbetween PSD#07. EHS covers for both PECs also opened. PYL station staff attended to assist the family. EHS covers were',
     'ATS Alarm manager showed Detrainment Door - DD Cover Open in PV37 Car1. Informed RS MCO Najib. Asset:8371/RSC/XDOR',
     'PV01/07: Rover reported train doors did not close with PSDs. Train departed MBT after PSDs recycled. Alarms displayed: Door Command Status - Door Command Failed. Stock changed at PYL. TSC Steven - 5\nPSD A2 informed\n\nMBT OT DEP DEV 00:00:45\n\nCCTV Playback confirm all train doors did not close initially. Train doors closed after PSDs recycled once.',
     'ATS Alarm Manager and TIP showed Train PV13 Car3: Interior smoke detection - Fault detected. Informed RS Tony. Asset : 8133/RSC/FSD.\t',
     'PV10, rover reported Car1 A2 did not opened at SER and LRC IT. All train doors and PSD opened except for Car1 A2 door. At MRM IT Car1 A2 door train door opened. Request rover to monitor for 3 stations, no abnormalities. Train stock changed at PYL. DSM informed.',
     'ATS Alarm Manager & TIP showed Train PV35 Car 3: Air Conditioning Return Air Dampers - Fails To Open. Informed RSMCO Siva. Asset: 8353/RSC/ACN/RDM/B1 8353/RSC/ACN/RDM/B2',
     'ATS Alarm Manager and TIP showed Train PV21 Car2: Interior smoke detection - Fault detected. Informed RS Najib. Asset : 8212/RSC/FSD.',
     'ATS Alarm Manager and TIP showed Train PV34 Car  1  Interior smoke detection - Fault detected.Informed RS\tYT Tan. Asset: 8341/RSC/FSD',
     'Rover report car2 neutral isolation 1 failed.',
     'ATS Alarm Manager and TIP showed Train PV24 Car 1: Air cooling function - Failed. Asset: 8241/RSC/ACN.\tInformed RS Tony.',
     'PV05 ATS alarm and TIP indicated  car 3 under seat fire detection under seat detected, rover on board confirmed no sight of smoke or fire, DSM informed.',
     'ATS alarm managers showed Door Open failure alarm and Ebrake by others.  Checked CCTV PSDs opened but train doors did not open.  SS activated to board train toopen train doors thorugh crew switch with ITAMA removed.  SS',
     'PV33 ATS alarm manager and TIP showed car 3 smoke fault.',
     'TIP shown CCTV ancillary major fault. Camera 7 black. Comms informed. Train door Car 3/B1 slow in closing.',
     'ATS Alarm Manager & TIP showed Train PV32 Car 2: Interior Smoke Detection - Fault Detected. InformedRSMCO Siva. Asset: 8322/RSC/FSD',
     'PV35 rover reported of TTIS showing Emergency  cover open please inform staff.  Check on IP page shows all status normal. Rover checked that all EHS is physically intact & No PA of Emergency cover  open were made in the']




```python
# use logistic regression
```


```python
# 1. import
from sklearn.linear_model import LogisticRegression

# 2. instantiate a logistic regression model
logreg = LogisticRegression()
```


```python
# 3. train the model using X_train_ft
logreg.fit(X_train_ft, y_train)
```

    C:\Users\Darius\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




    LogisticRegression()




```python
# 4. make class predictions for X_test_ft
y_pred_class = logreg.predict(X_test_ft)
```


```python
# calculate predicted probabilities for X_test_ft (well calibrated)
y_pred_prob = logreg.predict_proba(X_test_ft)[:,1]
y_pred_prob
```




    array([0.36460121, 0.00106865, 0.31593391, ..., 0.07327737, 0.85354075,
           0.04733495])




```python
# calculate accuracy
metrics.accuracy_score(y_test, y_pred_class)
```




    0.857953202552588




```python
# store the vocabulary of X_train
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)
```




    7194




```python
# examine the last 50 tokens
print(X_train_tokens[-50:])
```

    ['workshop', 'worn', 'worned', 'would', 'wound', 'wprking', 'written', 'wrong', 'wrongly', 'ws', 'wsp', 'wss', 'x2', 'x4', 'xdor', 'xx', 'y2', 'ya', 'yale', 'yalekey', 'yap', 'yapinfo', 'yapinformed', 'yapwas', 'yellow', 'yeo', 'yeow', 'yep', 'yet', 'yield', 'yielded', 'yong', 'york', 'yt', 'ytasset', 'ytinformed', 'yttan', 'yue', 'yup', 'zabbix', 'zabix', 'zaki', 'zebbix', 'zero', 'zeroed', 'zhang', 'zibbix', 'zone', 'zoneev13_4', 'zul']
    


```python
# Naive Bayes counts the number of times each token appears in each class
# trailing underscore - learned during fitting
nb.feature_count_
```




    array([[28.,  0.,  0., ..., 15.,  0.,  7.],
           [15.,  1.,  1., ...,  8.,  1.,  2.]])




```python
# rows represent classes, columns represent tokens
nb.feature_count_.shape
```




    (2, 7194)




```python
# number of times each token appears across all Critical messages
Critical_token_count = nb.feature_count_[1, :]
Critical_token_count
```




    array([15.,  1.,  1., ...,  8.,  1.,  2.])




```python
# number of times each token appears across all Non-Critical messages
Non_Critical_token_count = nb.feature_count_[0, :]
Non_Critical_token_count
```




    array([28.,  0.,  0., ..., 15.,  0.,  7.])




```python
# create a DataFrame of tokens with their separate critical and non-critical counts
tokens = pd.DataFrame({'token':X_train_tokens, 'Critical':Critical_token_count, 
                       'Non-Critical':Non_Critical_token_count}).set_index('token')
tokens.sample(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Critical</th>
      <th>Non-Critical</th>
    </tr>
    <tr>
      <th>token</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>68</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>circuitt0814</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1444hrs</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>3.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>andb2</th>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>doorc</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>about1metre</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>traininfo</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>going</th>
      <td>5.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>maunally</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>statusopened</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>wasloose</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>mps_ot</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ventilatiom</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2times</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>detrainmentdoor</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>trainpv10</th>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>decetected</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>wasa</th>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>andthe</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# examine 5 random DataFrame rows
# random_state=6 is a seed for reproducibility
tokens.sample(5, random_state=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Critical</th>
      <th>Non-Critical</th>
    </tr>
    <tr>
      <th>token</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pv03car</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>bright</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1243hrs</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>stains</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1had</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Naive Bayes counts the number of observations in each class
nb.class_count_
```




    array([8229., 4462.])




```python
tokens.dtypes
```




    Critical        float64
    Non-Critical    float64
    dtype: object




```python
# +1 to Critical and Non-Critical counts to avoid dividing by 0
tokens.iloc[:,0] = tokens.iloc[:,0].map(lambda x:x+1.0)
tokens.iloc[:,1] = tokens.iloc[:,1].map(lambda x:x+1.0)
tokens.sample(5, random_state=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Critical</th>
      <th>Non-Critical</th>
    </tr>
    <tr>
      <th>token</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pv03car</th>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>bright</th>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1243hrs</th>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>stains</th>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1had</th>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# convert the Critical and Non-Critical counts into frequencies
tokens['Critical_token_count'] = tokens["Critical"] / nb.class_count_[0]
tokens['Non_Critical_token_count'] = tokens["Non-Critical"] / nb.class_count_[1]
tokens.sample(5, random_state=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Critical</th>
      <th>Non-Critical</th>
      <th>Critical_token_count</th>
      <th>Non_Critical_token_count</th>
    </tr>
    <tr>
      <th>token</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pv03car</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.000243</td>
      <td>0.000224</td>
    </tr>
    <tr>
      <th>bright</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.000243</td>
      <td>0.000224</td>
    </tr>
    <tr>
      <th>1243hrs</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.000243</td>
      <td>0.000224</td>
    </tr>
    <tr>
      <th>stains</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.000122</td>
      <td>0.000448</td>
    </tr>
    <tr>
      <th>1had</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.000243</td>
      <td>0.000224</td>
    </tr>
  </tbody>
</table>
</div>




```python
# calculate the ratio of critical to non-critical for each token
tokens['Critical_ratio'] = tokens["Critical_token_count"]  / tokens["Non_Critical_token_count"]
tokens.sample(5, random_state=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Critical</th>
      <th>Non-Critical</th>
      <th>Critical_token_count</th>
      <th>Non_Critical_token_count</th>
      <th>Critical_ratio</th>
    </tr>
    <tr>
      <th>token</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pv03car</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.000243</td>
      <td>0.000224</td>
      <td>1.084457</td>
    </tr>
    <tr>
      <th>bright</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.000243</td>
      <td>0.000224</td>
      <td>1.084457</td>
    </tr>
    <tr>
      <th>1243hrs</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.000243</td>
      <td>0.000224</td>
      <td>1.084457</td>
    </tr>
    <tr>
      <th>stains</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.000122</td>
      <td>0.000448</td>
      <td>0.271114</td>
    </tr>
    <tr>
      <th>1had</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.000243</td>
      <td>0.000224</td>
      <td>1.084457</td>
    </tr>
  </tbody>
</table>
</div>




```python
# examine the DataFrame sorted by Critical_ratio. These are the words appeared that is most likely to belong to critical category.
tokens.loc[:,:].sort_values('Critical_ratio', ascending=False)[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Critical</th>
      <th>Non-Critical</th>
      <th>Critical_token_count</th>
      <th>Non_Critical_token_count</th>
      <th>Critical_ratio</th>
    </tr>
    <tr>
      <th>token</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>brk</th>
      <td>295.0</td>
      <td>4.0</td>
      <td>0.035849</td>
      <td>0.000896</td>
      <td>39.989367</td>
    </tr>
    <tr>
      <th>prop</th>
      <td>137.0</td>
      <td>3.0</td>
      <td>0.016648</td>
      <td>0.000672</td>
      <td>24.761777</td>
    </tr>
    <tr>
      <th>input</th>
      <td>83.0</td>
      <td>2.0</td>
      <td>0.010086</td>
      <td>0.000448</td>
      <td>22.502491</td>
    </tr>
    <tr>
      <th>module</th>
      <td>76.0</td>
      <td>2.0</td>
      <td>0.009236</td>
      <td>0.000448</td>
      <td>20.604691</td>
    </tr>
    <tr>
      <th>output</th>
      <td>74.0</td>
      <td>2.0</td>
      <td>0.008993</td>
      <td>0.000448</td>
      <td>20.062462</td>
    </tr>
    <tr>
      <th>bce</th>
      <td>600.0</td>
      <td>21.0</td>
      <td>0.072913</td>
      <td>0.004706</td>
      <td>15.492249</td>
    </tr>
    <tr>
      <th>ai2cb</th>
      <td>28.0</td>
      <td>1.0</td>
      <td>0.003403</td>
      <td>0.000224</td>
      <td>15.182404</td>
    </tr>
    <tr>
      <th>ai1cb</th>
      <td>26.0</td>
      <td>1.0</td>
      <td>0.003160</td>
      <td>0.000224</td>
      <td>14.097946</td>
    </tr>
    <tr>
      <th>propulsion</th>
      <td>764.0</td>
      <td>32.0</td>
      <td>0.092842</td>
      <td>0.007172</td>
      <td>12.945710</td>
    </tr>
    <tr>
      <th>fip</th>
      <td>70.0</td>
      <td>3.0</td>
      <td>0.008507</td>
      <td>0.000672</td>
      <td>12.652003</td>
    </tr>
  </tbody>
</table>
</div>




```python
# look up the spam_ratio for a given token
tokens.loc[:, 'Critical_ratio']
```




    token
    00            0.000227
    0001hrs       0.000000
    0002          0.000000
    0030hrs       0.000122
    0034hrs            inf
                    ...   
    zhang              inf
    zibbix             inf
    zone          0.000228
    zoneev13_4    0.000000
    zul           0.000425
    Name: Critical_ratio, Length: 7194, dtype: float64




```python
#Text Cleaning
```


```python
featuresNLP = features.copy()
```


```python
#1 change the text into lower case
featuresNLP['FDS_Remarks'] = featuresNLP['FDS_Remarks'].map(lambda x:x.lower() if type(x) == str else x)
```


```python
#2 remove punctuation using regex
featuresNLP['FDS_Remarks'] = featuresNLP['FDS_Remarks'].str.replace('[^\w\s]','', regex=True)
freq1 = pd.Series(' '.join(featuresNLP['FDS_Remarks']).split()).value_counts()
freq1
```




    and                     12008
    train                   11066
    alarm                   10754
    at                      10563
    informed                10419
                            ...  
    8303rscacnfdmvb2            1
    8303rscacnfdmva2            1
    8303rscacnacnedmv           1
    najibasset8303rscacn        1
    pv6374                      1
    Length: 13071, dtype: int64




```python
#3 remove number using regex
featuresNLP['FDS_Remarks'] = featuresNLP['FDS_Remarks'].str.replace('\d','', regex=True)
```


```python
#4 removal of stop words
stop = stopwords.words('english')

featuresNLP['FDS_Remarks'] = featuresNLP['FDS_Remarks'].map(lambda x: " ".join(x for x in x.split() if x not in stop))
freq2 = pd.Series(' '.join(featuresNLP['FDS_Remarks']).split()).value_counts()
freq2
```




    pv                 15482
    car                14907
    train              11102
    alarm              10755
    informed           10481
                       ...  
    biju                   1
    instancesof            1
    faultinformedrs        1
    requiredcleaner        1
    inforrmed              1
    Length: 7363, dtype: int64




```python
#5 removal of single alphabet words
featuresNLP['FDS_Remarks'] = featuresNLP['FDS_Remarks'].map(lambda x: " ".join(x for x in x.split() if len(x)>1))
freq3 = pd.Series(' '.join(featuresNLP['FDS_Remarks']).split()).value_counts()
freq3
```




    pv                 15482
    car                14907
    train              11102
    alarm              10755
    informed           10481
                       ...  
    burst                  1
    thatprerecorded        1
    msg                    1
    haifd                  1
    inforrmed              1
    Length: 7346, dtype: int64




```python
#6 lemmnization (better than stemming)
lm = WordNetLemmatizer()

featuresNLP['FDS_Remarks'] = featuresNLP['FDS_Remarks'][:].apply(lambda x: " ".join([lm.lemmatize(word) for word in x.split()]))
freq4 = pd.Series(' '.join(featuresNLP['FDS_Remarks']).split()).value_counts()
freq4
```




    car            15732
    pv             15482
    alarm          11659
    train          11215
    informed       10481
                   ...  
    fluactuated        1
    breturn            1
    dampe              1
    vibrationin        1
    inforrmed          1
    Length: 7151, dtype: int64




```python
#7 removal top 10 frequent words
freqwords = pd.Series(' '.join(featuresNLP['FDS_Remarks']).split()).value_counts()[:10]
featuresNLP['FDS_Remarks'] = featuresNLP['FDS_Remarks'].map(lambda x: " ".join(x for x in x.split() if x not in freqwords))
freq5 = pd.Series(' '.join(featuresNLP['FDS_Remarks']).split()).value_counts()
freq5
```




    r                  4760
    air                4709
    asset              4256
    rover              3910
    fault              3852
                       ... 
    servive               1
    theextinguisher       1
    rotary                1
    machinery             1
    inforrmed             1
    Length: 7141, dtype: int64




```python
#8 removal words appear less than 20 times. removed ~90% of the words
freqwords = pd.Series(' '.join(featuresNLP['FDS_Remarks']).split()).value_counts()[-6368:]
featuresNLP['FDS_Remarks'] = featuresNLP['FDS_Remarks'].map(lambda x: " ".join(x for x in x.split() if x not in freqwords))
freq6 = pd.Series(' '.join(featuresNLP['FDS_Remarks']).split()).value_counts()
freq6
```




    r              4760
    air            4709
    asset          4256
    rover          3910
    fault          3852
                   ... 
    binformed        21
    alight           21
    outstanding      21
    attempt          21
    issued           21
    Length: 773, dtype: int64




```python
featuresNLP
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FDS_Remarks</th>
      <th>Criticality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>brake system shown bce minor propulsion isolat...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ttis display door closing door psds closing bl...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>door ehs activation status keep unknown status</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cooling cso site cooling system ok</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cso rover reported ttis showing station rsmco</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16917</th>
      <td>rover reported saloon light near door lighted dsm</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16918</th>
      <td>rover reported salon lighting working door dsm</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16919</th>
      <td>eb atp itama removed departing bsh pax exchang...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16920</th>
      <td>staff feedback unusual noise rover checked con...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16921</th>
      <td>atc internal comms failure total comms failure...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>16922 rows × 2 columns</p>
</div>




```python
#using naives
X = featuresNLP.FDS_Remarks
y = featuresNLP.Criticality
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_train_ft = vect.fit_transform(X_train)
X_test_ft = vect.transform(X_test)
nb.fit(X_train_ft, y_train)
y_pred_class = nb.predict(X_test_ft)
metrics.accuracy_score(y_test, y_pred_class)
```




    0.8329000236350744




```python
#using logisticregression
logreg.fit(X_train_ft, y_train)
y_pred_class = logreg.predict(X_test_ft)
y_pred_prob = logreg.predict_proba(X_test_ft)[:,1]
metrics.accuracy_score(y_test, y_pred_class)
```

    C:\Users\Darius\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




    0.8553533443630348




```python

```
