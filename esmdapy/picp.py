import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import norm


def PICP(posterior, data, CI):
    final = pd.read_csv(posterior)
    final.set_index('Unnamed: 0', inplace=True)
    observed = pd.read_csv(data)
    
    upper_ci = (100-(100-CI)/2)/100
    lower_ci = ((100-CI)/2)/100
    
    upper = final.quantile(upper_ci, axis=1)
    lower = final.quantile(lower_ci, axis=1)
    
    included = 0
    for i in range(len(observed)):
        if (observed.value[i] >= lower[i]) & (observed.value[i] <= upper[i]):
            included +=1
    PP = included/len(observed)*100
    return PP


def PICP2(posterior, data, CI): 
    data_in = pd.read_csv(data)
    data_out = pd.read_csv(posterior)
    lower = (50-CI/2)/100
    upper = (50+CI/2)/100
    data_set = data_in.copy() 
    data_set['lower_quant'] = data_out.quantile(lower, axis=1)
    data_set['upper_quant'] = data_out.quantile(upper, axis=1)
    data_set['overlap'] = norm.cdf(data_set['upper_quant'],loc=data_set['value'],scale=data_set['noise']) - norm.cdf(data_set['lower_quant'],loc=data_set['value'],scale=data_set['noise'])
    score = data_set['overlap'].sum()/len(data_set)*100
    return score