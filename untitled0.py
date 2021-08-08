



import pandas as pd

trainy = pd.read_csv(r'C:\Users\Odd\Desktop\New folder (2)\TRAIN.csv')
testy = pd.read_csv(r'C:\Users\Odd\Desktop\New folder (2)\TEST.csv')

train = trainy.dropna()

genderdummy = pd.get_dummies(train, columns = ['Self_Employed', 'Loan_Status', 'Gender'])


x3z = genderdummy.drop('Loan_ID',axis = 1)
x3 = x3z.drop('Dependents',axis = 1)
x2 = x3.drop('Loan_Status_N', axis = 1)
x3p = x2.drop('Loan_Status_Y', axis = 1)
xxz2 = x3p.drop('Married', axis=1)
xxz1 = xxz2.drop('Education', axis=1)
xxz = xxz1.drop('Property_Area',axis=1)




y = genderdummy['Loan_Status_Y']

test = testy.fillna(0, inplace = True)
genderdummyt = pd.get_dummies(testy, columns = ['Self_Employed', 'Gender'])

x3xy = genderdummyt.drop('Loan_ID', axis = 1)
x3x =  x3xy.drop('Dependents', axis = 1)
ss = x3x.drop('Gender_0', axis=1)
xx3 = ss.drop('Self_Employed_0', axis = 1)
xxz22 = xx3.drop('Married', axis=1)
xxz11 = xxz22.drop('Education', axis=1)

xxa = xxz11.drop('Property_Area', axis=1)

from sklearn.tree import DecisionTreeClassifier
print(xxz.info())
print(xxa.info())
k = DecisionTreeClassifier(criterion='entropy', random_state = 0)
w = k.fit(xxz,y)
prdbro = w.predict(xxa)






           









