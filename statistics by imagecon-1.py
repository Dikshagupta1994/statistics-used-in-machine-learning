from scipy.stats import norm
import seaborn as sns
import numpy as np

#first question
from scipy.stats import poisson
cdfp=poisson.cdf(15,mu=10)
prob=1-cdfp
print(prob)


#2nd question
cdfn=norm(loc=10,scale=2).cdf(5)
print(cdfn)

#3rd question
cdfp1=poisson.cdf(200,mu=100)
prob1=1-cdfp1
print(prob1)

#4th question
import pandas as pd
data={'Subject Name':
      ['english','hindi','sanskrit','math','science','social'],
      'Total Marks':[50,50,50,50,50,50],
      'obtained marks':[40,41,33,34,32,34]}
df=pd.DataFrame(data)
df.mean()
df.median()
df.mode()
df.var()
df.std()












