import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle

df = pd.read_csv("glassdoor_data_cleaned.csv")
df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided',
             'job_state','same_state','company_age','python','spark','aws','excel','job_simp','seniority','desc_len']]
#Getting dummy data
df_dummy = pd.get_dummies(df_model)

X = df_dummy.drop('avg_salary', axis=1)
Y = df_dummy.avg_salary.values

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=42)

rf = RandomForestRegressor()
print("GridSearchCV started...")
parameters = {
    'n_estimators': range(10,300,10),
    'criterion': ('squared_error', 'absolute_error'),
    'max_features': ('auto', 'sqrt', 'log2')
}
gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_Train, Y_Train)

print("gs.best_estimator_: ", gs.best_estimator_)

pickl = {'model': gs.best_estimator_}
#wb = write binary
pickle.dump( pickl, open( 'model_file.p', "wb" ) )

file_name = "model_file.p"
#rb = read binary
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

print("Predicting...")
predicted_value = model.predict(np.array(list(X_Test.iloc[1,:])).reshape(1,-1))[0]
print("Predicted average salary: ", predicted_value)
