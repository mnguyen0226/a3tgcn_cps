# Scripts takes in the labels and prediction dataset to calculate the mahalanobis distance
# Test calculate the first 10 rows
# Test calculate the first 20 rows
# Calculate and return if there is any negative numbers
# Check the math for robust mahalanobis distance


from dis import dis
import pandas as pd
import numpy as np
from scipy.stats import chi2
import csv

def data_preprocessing(num_line = 9):
    df_eval_labels = []
    df_eval_preds = []
    
    # Read in the eval ground truth  
    with open('out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/eval/test_labels.csv') as label_file:
        
        # Read the csv file
        csv_file = csv.reader(label_file)
        
        # Print the first 10 lines of the csv file
        for i, line in enumerate(csv_file):
            df_eval_labels.append(line)
            if(i >= num_line):
                break
    # Close ground truth file
    label_file.close()
    
    # Read in the eval prediction
    with open('out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/eval/test_output.csv') as pred_file:
        
        # Read tge csv file
        csv_file = csv.reader(pred_file)
        
        # Print the first 10 lines of the csv file
        for i, line in enumerate(csv_file):
            df_eval_preds.append(line)
            if(i >= num_line):
                break
    # Close prediction file
    pred_file.close()
    
    # Converts variable from string to float
    for i in range (len(df_eval_labels)):
        df_eval_labels[i] = [float(x) for x in df_eval_labels[i]]
    for i in range (len(df_eval_preds)):
        df_eval_preds[i] = [float(x) for x in df_eval_preds[i]]
    
    return df_eval_labels, df_eval_preds


def calculate_md():
    # Get lists
    df_eval_labels, df_eval_preds = data_preprocessing(num_line=1743)
    
    # Convert lists to numpy arrays
    df_eval_labels = np.array(df_eval_labels)
    df_eval_preds = np.array(df_eval_preds)
    
    # Get error array
    df_error = df_eval_labels - df_eval_preds

    # 1. Calculate the covariance matrix
    cov = np.cov(df_error, rowvar=False)
    
    # 2. Calculate cov^-1
    covariance_pm1 = np.linalg.matrix_power(cov, -1)
    
    # 3. Calculate the mean error arrayy
    mean_error = np.mean(df_error, axis=0)
    
    # 4. Calculate the mahalanobis distance
    distances = []
    for i, val in enumerate(df_error):
        p1 = val  # Ozone and Temp of the ith row
        p2 = mean_error
        distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
        distances.append(distance)
        print(f"Distance: {distance}")
    distances = np.array(distances)

    # print(distances)
    
    nega = [distances[i] for i in range (len(distances)) if distances[i] <= 0.0]
    print(nega)
    
if __name__ == "__main__":
    calculate_md()