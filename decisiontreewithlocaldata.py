import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
df=pd.read_csv("decisiontree.csv")
df.head()
print(df)
inputs=df.drop('salary_more_then_10k', axis='columns')
target=df['salary_more_then_10k']
print(target)
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()
inputs['company_n']=le_company.fit_transform(inputs['campany'])
inputs['job_n']=le_job.fit_transform(inputs['job'])
inputs['degree_n']=le_degree.fit_transform(inputs['degree'])
inputs.head()
print(inputs)

inputs_n=inputs.drop(['campany','job','degree'], axis='columns')
print(inputs_n)

model=tree.DecisionTreeClassifier() # model training
model.fit(inputs_n,target)

pred1=model.predict([[0,0,2,0]])
print("Prediction of google     sales executive  bachlors is :",pred1)
accuracy=model.score(inputs_n, target)
print(f'Accuracy:{accuracy*100:.2f}%')