import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn import linear_model




def colum_names():
    col_names = ['Entity',
             'Code',
             'Year',
             'Prevalence - Neoplasms - Sex: Both - Age: 70+ years (Number)',
             'Prevalence - Neoplasms - Sex: Both - Age: 50-69 years (Number)',
             'Prevalence - Neoplasms - Sex: Both - Age: 15-49 years (Number)',
             'Prevalence - Neoplasms - Sex: Both - Age: 5-14 years (Number)',
             'Prevalence - Neoplasms - Sex: Both - Age: Under 5 (Number)']
    
    return col_names

def get_csv_people_cancer_by_age():
    col_names = colum_names()
    number_people_cancer_by_age = pd.read_csv(
        'dataset/number-of-people-with-cancer-by-age.csv', header=None, names=col_names)
    return number_people_cancer_by_age


people_cancer_by_age = get_csv_people_cancer_by_age()

X = people_cancer_by_age.drop('Year', axis='columns') 
y = people_cancer_by_age.Year

reg = linear_model.LinearRegression()
reg.fit(X, y)

print('Prediccion # de celdas', reg.predict([[2.3]]) )