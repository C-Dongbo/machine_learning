import numpy as np
import pandas as pd
import pyspark
from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

sc = SparkContext(master="local", appName="first app")

df_rdd = sc.textFile('./data/ml-1m/ratings.dat').map(lambda x: x.split("::"))

ratings= df_rdd.map(lambda l : Rating(int(l[0]),int(l[1]),float(l[2]))) 
X_train, X_test= ratings.randomSplit([0.8, 0.2])

rank = 10
numIterations = 10
model = ALS.train(X_train, rank, numIterations)

testdata = X_test.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))

ratesAndPreds = X_test.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)

MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))