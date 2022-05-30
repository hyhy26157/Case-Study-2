
# Using Reported Fault Symptoms to Predict Safety Related Faults.

## Motivation for this study
Usually, when a defect is reported during train servicing, the defect symptoms will be reported and the technician will follow up on the troubleshooting. Fault group classification is provided only when troubleshooting has completed. An experienced planner will prioritise the defect accordingly, based on the symptoms, if it is a safety related delay. However, this process can be automated using machine learning algorithms, such as the Navies Baynes or the Logistic Regression which are a popular ML model for sentiment analysis, to predict if the fault is safety related. This will reduce manual labeling of the fault type. 

## Data 
Data showcase in this case study are gathered from a train operator based in a highly populated and warm area. Reported faults are between 2010-Jan to 2020-Dec. Sensitive information has been replaced with a dummy information.

There are 5 columns

 1. Record_Date - Date of the reported fault    
 2. Activity_Type - Type of Fault Activity. Activity Type 3 is corrective maintenance (CM) fault. Remaining Type are non-CM faults.        
 3. Service_Number - reported fault service ID. If there is no service number. The reported fault found during non-service hours. 
 4. Service_Fault_Symptoms - reported fault symptoms text
 5. Fault_Group - Classification of fault. Will be using this to seperate between safety and non-safety faults.

## Step

Steps involved in this study include

1. Cleaning of data - removal of unncessary data and filling Nil data - Using Pandas library.
2. Explortary Data Analysis - explore interesting trend in the data - Using matlibplot and seaborn library.
3. Machine Learning - using 2 popular ML model (Naive Bayes model and Logistic Regression)
4. Text Analysis with Feature Engineering - create feature data to look at the most popular word in reported symptoms that has highest correlationship with safety related defects.
5. Natural Language Processing - using popular NLP technique to reduce noises in the service fault symptoms text to improve the ML.

## Summary

The ML model is able to predict the fault type ( Safety related or non safety related) using fault symptoms with 90.3% accuracy for NB model and 93.9% accuracy with LR model. The model is further improved with NLP to 92.5% accuracy for MB model but accuary remains the same for LR model.

Other form of analysis that can be done to further improve the ML model.

1) To further analyse symptoms that has tagged as False-Positive and False-Negative.
2) To manually recategorise misc fault category symptoms.

