import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

df = pd.read_csv("canada_per_capita_income.csv")
print(df.head())

plt.xlabel("year(x)")
plt.ylabel("per capital income(USD)")
plt.scatter(df.year,df.percapitalincome)
plt. show()

reg = linear_model.LinearRegression()
print(reg.fit(df[["year"]],df.percapitalincome))
print(reg.predict([[2020]]))

print(reg.coef_)

print(reg.intercept_)

print(2020*828.46507522 - 1632210.7578554575)

plt.xlabel("year(sqr ft)")
plt.ylabel("per capital income(USD)")
plt.scatter(df.year,df.percapitalincome)
plt.plot(df.year,reg.predict(df[["year"]]),color="blue")
plt. show()

d = pd.read_csv("year.csv")
print(d)
print(reg.predict(d))

p = reg.predict(d)
d['per capital income'] = p
print(d)

d.to_csv("p2.csv")
