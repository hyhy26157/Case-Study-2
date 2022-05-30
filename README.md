
# Case Study 2: Using Reported Fault Symptoms to Predict Safety Related Faults.

## Motivation for this study
Currently, fault group classification is provided only when troubleshooting has completed. However, service fault symptoms provides an indication if fault is safety related which allows planner to improve work prioritisation. 

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

