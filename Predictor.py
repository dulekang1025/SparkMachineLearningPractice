from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator


def predictor (model, grid, training, testing, model_name):
    # 4.3 evaluator
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label_index')

    # 4.4 cross validation
    cv = CrossValidator(estimator=model, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5)
    model = cv.fit(training)

    # 4.5 prediction
    prediction = model.transform(testing).cache()
    # prediction.show(10)

    # 5.1 evaluation
    accuracy = prediction.filter(prediction.label_index == prediction.prediction).count() / prediction.count()
    print(model_name + ' accuracy: ' + str(accuracy))


