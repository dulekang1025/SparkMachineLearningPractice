from pyspark import SparkContext
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import mean, max, regexp_replace
from pyspark.ml import Pipeline
from Predictor import predictor

spark = SparkSession.builder.getOrCreate()
df = spark.read.options(header="true", inferschema="true").csv('Data/mushrooms.csv')

# 0. drop column veil-type
df = df.drop('veil-type')

# 1.1 fill empty value, (replace ? with max value of that column)
print('Total missing value in column stalk-root: ' + str(df.filter(df['stalk-root'] == '?').count()))
max = df.select(max(df['stalk-root'])).collect()
df = df.withColumn('stalk-root', regexp_replace('stalk-root', '\\?', max[0][0]))

# 1.2 or just drop instances (rows) with value '?' in it
# print('Dropped missing values in column stalk-root.')
# df = df.filter(df['stalk-root'] != '?')

# 2.1 use one hot encoding
feature_cols = df.schema.names[1:]
label = df.schema.names[0]
# string indexer
feature_indexers = [StringIndexer(inputCol=col, outputCol=col+'_index') for col in feature_cols]
label_indexer = StringIndexer(inputCol=label, outputCol='label_index')
# one hot encoder
encoders =[OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=indexer.getOutputCol() + '_1hot')
          for indexer in feature_indexers]
# vector assembler
assembler = VectorAssembler(inputCols=[col.getOutputCol() for col in encoders], outputCol='features')
# pipeline
pipeline = Pipeline(stages=feature_indexers + encoders + [assembler] + [label_indexer])
# transform
df_trans = pipeline.fit(df).transform(df).cache()
# df_trans.show()


# 2.2 dummy encoding
# feature_cols = df.schema.names[1:]
# label = df.schema.names[0]
# # string indexer
# feature_indexers = [StringIndexer(inputCol=col, outputCol=col+'_index') for col in feature_cols]
# label_indexer = StringIndexer(inputCol=label, outputCol='label_index')
# assembler = VectorAssembler(inputCols=[col.getOutputCol() for col in feature_indexers], outputCol='features')
# # pipeline
# pipeline = Pipeline(stages=feature_indexers + [assembler] + [label_indexer])
# # transform
# df_trans = pipeline.fit(df).transform(df).cache()
# # df_trans.show(10)

# 3.1 split training and testing set
(training, testing) = df_trans.randomSplit([0.7, 0.3], seed=12345)

# 4.1 train the RF model
rf_model = RandomForestClassifier(labelCol='label_index', featuresCol='features')

# 4.2 rf parameter grid
# rf_grid = ParamGridBuilder().addGrid(rf_model.maxDepth, [4, 5, 6])\
#     .addGrid(rf_model.numTrees, [50, 75, 100])\
#     .addGrid(rf_model.maxBins, [15, 20, 25]).build()

rf_grid = ParamGridBuilder().addGrid(rf_model.maxDepth, [4, 5, 6])\
    .addGrid(rf_model.numTrees, [100]).build()

# 4.3 predict
predictor(rf_model, rf_grid, training, testing, 'Random Forest Classification')

# 5.1 train the NB model
nb_model = NaiveBayes(labelCol='label_index', featuresCol='features')

# 5.2 nb parameter grid
nb_grid = ParamGridBuilder().addGrid(nb_model.smoothing, [0.0, 0.4, 0.8, 1.0]).build()

# 5.3 predict
predictor(nb_model, nb_grid, training, testing, 'Naive Bayes Classification')

