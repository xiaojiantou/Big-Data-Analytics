Linear Support Vector Machines (SVMs)

from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint


def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])


data = sc.textFile("/Users/arrowlittle/Desktop/data/pima-indians-diabetes.data.txt")


parsedData = data.map(parsePoint)
model1 = SVMWithSGD.train(parsedData, iterations=10)

labelsAndPreds = parsedData.map(lambda p: (p.label, model1.predict(p.features)))

trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())

print("Training Error = " + str(trainErr))

model1.save(sc, "/Users/arrowlittle/Desktop/data/australianSVMWithSGDModel")


Linear least squares, Lasso, and ridge regression

from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

from pyspark.mllib.regression import LabeledPoint
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])


data = sc.textFile("/Users/arrowlittle/Desktop/data/pima-indians-diabetes.data.txt")


parsedData = data.map(parsePoint)
model2 = LinearRegressionWithSGD.train(parsedData, iterations=100, step=0.00000001)

valuesAndPreds = parsedData.map(lambda p: (p.label, model2.predict(p.features)))
MSE = valuesAndPreds \
    .map(lambda (v, p): (v - p)**2) \
    .reduce(lambda x, y: x + y) / valuesAndPreds.count()
print("Mean Squared Error = " + str(MSE))

model2.save(sc, "/Users/arrowlittle/Desktop/data/australianLinearRegressionWithSGDModel")