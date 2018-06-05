#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 01:04:43 2017

@author: arrowlittle
"""

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

lines2=spark.read.text("/Users/arrowlittle/Desktop/data/ml-latest-small/ratings.csv").rdd

header=lines2.first()
lines2=lines2.filter(lambda row: row!=header) 
parts2=lines2.map(lambda row: row.value.split(",")) 

ratingsRDD2 = parts2.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=int(p[3])))

ratings2 = spark.createDataFrame(ratingsRDD2)
ratings2.show()

(training, test) = ratings2.randomSplit([0.8, 0.2])
als = ALS(maxIter=5, regParam=0.01, implicitPrefs=True,userCol="userId", 
          itemCol="movieId", ratingCol="rating",coldStartStrategy="drop")
model = als.fit(training)

predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

userRecs = model.recommendForAllUsers(10)

movieRecs = model.recommendForAllItems(10)

userRecs.show(5,False)