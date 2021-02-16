// Databricks notebook source
// MAGIC %md
// MAGIC # Machine Learning With Spark ML
// MAGIC In this lab assignment, you will complete a project by going through the following steps:
// MAGIC 1. Get the data.
// MAGIC 2. Discover the data to gain insights.
// MAGIC 3. Prepare the data for Machine Learning algorithms.
// MAGIC 4. Select a model and train it.
// MAGIC 5. Fine-tune your model.
// MAGIC 6. Present your solution.
// MAGIC 
// MAGIC As a dataset, we use the California Housing Prices dataset from the StatLib repository. This dataset was based on data from the 1990 California census. The dataset has the following columns
// MAGIC 1. `longitude`: a measure of how far west a house is (a higher value is farther west)
// MAGIC 2. `latitude`: a measure of how far north a house is (a higher value is farther north)
// MAGIC 3. `housing_,median_age`: median age of a house within a block (a lower number is a newer building)
// MAGIC 4. `total_rooms`: total number of rooms within a block
// MAGIC 5. `total_bedrooms`: total number of bedrooms within a block
// MAGIC 6. `population`: total number of people residing within a block
// MAGIC 7. `households`: total number of households, a group of people residing within a home unit, for a block
// MAGIC 8. `median_income`: median income for households within a block of houses
// MAGIC 9. `median_house_value`: median house value for households within a block
// MAGIC 10. `ocean_proximity`: location of the house w.r.t ocean/sea

// COMMAND ----------

// MAGIC %md
// MAGIC # 1. Get the data
// MAGIC Let's start the lab by loading the dataset. The can find the dataset at `data/housing.csv`. To infer column types automatically, when you are reading the file, you need to set `inferSchema` to true. Moreover enable the `header` option to read the columns' name from the file.

// COMMAND ----------

val housing = spark
  .read
  .format("csv")
  .option("sep",",")
  .option("inferSchema", "true")
  .option("header", "true")
  .load("dbfs:/FileStore/shared_uploads/angel@igareta.com/housing.csv")

// COMMAND ----------

// MAGIC %md
// MAGIC # 2. Discover the data to gain insights
// MAGIC Now it is time to take a look at the data. In this step we are going to take a look at the data a few different ways:
// MAGIC * See the schema and dimension of the dataset
// MAGIC * Look at the data itself
// MAGIC * Statistical summary of the attributes
// MAGIC * Breakdown of the data by the categorical attribute variable
// MAGIC * Find the correlation among different attributes
// MAGIC * Make new attributes by combining existing attributes

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.1. Schema and dimension
// MAGIC Print the schema of the dataset

// COMMAND ----------

housing.schema

// COMMAND ----------

// MAGIC %md
// MAGIC Print the number of records in the dataset.

// COMMAND ----------

housing.count()

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.2. Look at the data
// MAGIC Print the first five records of the dataset.

// COMMAND ----------

housing.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC Print the number of records with population more than 10000.

// COMMAND ----------

housing.where("population > 10000").count()

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.3. Statistical summary
// MAGIC Print a summary of the table statistics for the attributes `housing_median_age`, `total_rooms`, `median_house_value`, and `population`. You can use the `describe` command.

// COMMAND ----------

housing.describe("housing_median_age", "total_rooms", "median_house_value", "population").show()

// COMMAND ----------

// MAGIC %md
// MAGIC Print the maximum age (`housing_median_age`), the minimum number of rooms (`total_rooms`), and the average of house values (`median_house_value`).

// COMMAND ----------

import org.apache.spark.sql.functions._

housing.select(max("housing_median_age"), min("total_rooms"), avg("median_house_value")).show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.4. Breakdown the data by categorical data
// MAGIC Print the number of houses in different areas (`ocean_proximity`), and sort them in descending order.

// COMMAND ----------

housing.groupBy("ocean_proximity").count().sort(desc("count")).show()

// COMMAND ----------

// MAGIC %md
// MAGIC Print the average value of the houses (`median_house_value`) in different areas (`ocean_proximity`), and call the new column `avg_value` when print it.

// COMMAND ----------

housing.groupBy("ocean_proximity").agg(avg("median_house_value").as("avg_value")).show()

// COMMAND ----------

// MAGIC %md
// MAGIC Rewrite the above question in SQL.

// COMMAND ----------

housing.createOrReplaceTempView("df")
spark.sql("select ocean_proximity, avg(median_house_value) as avg_value from df group by ocean_proximity").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.5. Correlation among attributes
// MAGIC Print the correlation among the attributes `housing_median_age`, `total_rooms`, `median_house_value`, and `population`. To do so, first you need to put these attributes into one vector. Then, compute the standard correlation coefficient (Pearson) between every pair of attributes in this new vector. To make a vector of these attributes, you can use the `VectorAssembler` Transformer.

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler

val va = new VectorAssembler().setInputCols(Array("housing_median_age", "total_rooms", "median_house_value", "population")).setOutputCol("info")

val housingAttrs = va.transform(housing)

housingAttrs.show(5)

// COMMAND ----------

import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row

val Row(coeff: Matrix) = Correlation.corr(housingAttrs, "info").head
println(s"The standard correlation coefficient:\n ${coeff}")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.6. Combine and make new attributes
// MAGIC Now, let's try out various attribute combinations. In the given dataset, the total number of rooms in a block is not very useful, if we don't know how many households there are. What we really want is the number of rooms per household. Similarly, the total number of bedrooms by itself is not very useful, and we want to compare it to the number of rooms. And the population per household seems like also an interesting attribute combination to look at. To do so, add the three new columns to the dataset as below. We will call the new dataset the `housingExtra`.
// MAGIC ```
// MAGIC rooms_per_household = total_rooms / households
// MAGIC bedrooms_per_room = total_bedrooms / total_rooms
// MAGIC population_per_household = population / households
// MAGIC ```

// COMMAND ----------

val housingCol1 = housing.withColumn("rooms_per_household", expr("total_rooms/households"))
val housingCol2 = housingCol1.withColumn("bedrooms_per_room", expr("total_bedrooms/total_rooms"))
val housingExtra = housingCol2.withColumn("population_per_household", expr("population/households"))

housingExtra.select("rooms_per_household", "bedrooms_per_room", "population_per_household").show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC # 3. Prepare the data for Machine Learning algorithms
// MAGIC Before going through the Machine Learning steps, let's first rename the label column from `median_house_value` to `label`.

// COMMAND ----------

val renamedHousing = housingExtra.withColumnRenamed("median_house_value", "label")

// COMMAND ----------

// MAGIC %md
// MAGIC Now, we want to separate the numerical attributes from the categorical attribute (`ocean_proximity`) and keep their column names in two different lists. Moreover, sice we don't want to apply the same transformations to the predictors (features) and the label, we should remove the label attribute from the list of predictors. 

// COMMAND ----------

// label columns
val colLabel = "label"

// categorical columns
val colCat = "ocean_proximity"

// numerical columns
val colNum = renamedHousing.columns.filter(_ != colLabel).filter(_ != colCat)

// COMMAND ----------

// MAGIC %md
// MAGIC ## 3.1. Prepare continuse attributes
// MAGIC ### Data cleaning
// MAGIC Most Machine Learning algorithms cannot work with missing features, so we should take care of them. As a first step, let's find the columns with missing values in the numerical attributes. To do so, we can print the number of missing values of each continues attributes, listed in `colNum`.

// COMMAND ----------

for (c <- colNum) {
    renamedHousing.select(count(when(col(c).isNull, c)).alias(c)).show()
}

// COMMAND ----------

// MAGIC %md
// MAGIC As we observerd above, the `total_bedrooms` and `bedrooms_per_room` attributes have some missing values. One way to take care of missing values is to use the `Imputer` Transformer, which completes missing values in a dataset, either using the mean or the median of the columns in which the missing values are located. To use it, you need to create an `Imputer` instance, specifying that you want to replace each attribute's missing values with the "median" of that attribute.

// COMMAND ----------

import org.apache.spark.ml.feature.Imputer

val imputer = new Imputer()
    .setStrategy("median")
    .setInputCols(Array("total_bedrooms", "bedrooms_per_room"))
    .setOutputCols(Array("total_bedrooms", "bedrooms_per_room"))          
val imputedHousing = imputer.fit(renamedHousing).transform(renamedHousing)

imputedHousing.select("total_bedrooms", "bedrooms_per_room").show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Scaling
// MAGIC One of the most important transformations you need to apply to your data is feature scaling. With few exceptions, Machine Learning algorithms don't perform well when the input numerical attributes have very different scales. This is the case for the housing data: the total number of rooms ranges from about 6 to 39,320, while the median incomes only range from 0 to 15. Note that scaling the label attribues is generally not required.
// MAGIC 
// MAGIC One way to get all attributes to have the same scale is to use standardization. In standardization, for each value, first it subtracts the mean value (so standardized values always have a zero mean), and then it divides by the variance so that the resulting distribution has unit variance. To do this, we can use the `StandardScaler` Estimator. To use `StandardScaler`, again we need to convert all the numerical attributes into a big vectore of features using `VectorAssembler`, and then call `StandardScaler` on that vactor.

// COMMAND ----------

import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}

val va = new VectorAssembler()
    .setInputCols(colNum)
    .setOutputCol("attributes")
val featuredHousing = va.transform(imputedHousing)

//should i also put .setWithStd() and .setWithMean()?
val scaler = new StandardScaler()
    .setInputCol("attributes")
    .setOutputCol("scaled_attributes")
val scaledHousing = scaler.fit(featuredHousing).transform(featuredHousing)

scaledHousing.select("scaled_attributes").show(5, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## 3.2. Prepare categorical attributes
// MAGIC After imputing and scaling the continuse attributes, we should take care of the categorical attributes. Let's first print the number of distict values of the categirical attribute `ocean_proximity`.

// COMMAND ----------

renamedHousing.select(countDistinct(colCat)).show

// COMMAND ----------

// MAGIC %md
// MAGIC ### String indexer
// MAGIC Most Machine Learning algorithms prefer to work with numbers. So let's convert the categorical attribute `ocean_proximity` to numbers. To do so, we can use the `StringIndexer` that encodes a string column of labels to a column of label indices. The indices are in [0, numLabels), ordered by label frequencies, so the most frequent label gets index 0.

// COMMAND ----------

import org.apache.spark.ml.feature.StringIndexer

val indexer = new StringIndexer()
    .setInputCol(colCat)
    .setOutputCol("category_index")
val idxHousing = indexer.fit(renamedHousing).transform(renamedHousing)

idxHousing.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC Now we can use this numerical data in any Machine Learning algorithm. You can look at the mapping that this encoder has learned using the `labels` method: "<1H OCEAN" is mapped to 0, "INLAND" is mapped to 1, etc.

// COMMAND ----------

indexer.fit(renamedHousing).labelsArray

// COMMAND ----------

// MAGIC %md
// MAGIC ### One-hot encoding
// MAGIC Now, convert the label indices built in the last step into one-hot vectors. To do this, you can take advantage of the `OneHotEncoderEstimator` Estimator.

// COMMAND ----------

idxHousing.select("category_index").show(55, false)

// COMMAND ----------

import org.apache.spark.ml.feature.OneHotEncoder

val encoder = new OneHotEncoder()
    .setInputCol("category_index")
    .setOutputCol("category_index_vector")
val ohHousing = encoder.fit(idxHousing).transform(idxHousing)

ohHousing.select("category_index", "category_index_vector").show(55, false)

// COMMAND ----------

// MAGIC %md
// MAGIC # 4. Pipeline
// MAGIC As you can see, there are many data transformation steps that need to be executed in the right order. For example, you called the `Imputer`, `VectorAssembler`, and `StandardScaler` from left to right. However, we can use the `Pipeline` class to define a sequence of Transformers/Estimators, and run them in order. A `Pipeline` is an `Estimator`, thus, after a Pipeline's `fit()` method runs, it produces a `PipelineModel`, which is a `Transformer`.
// MAGIC 
// MAGIC Now, let's create a pipeline called `numPipeline` to call the numerical transformers you built above (`imputer`, `va`, and `scaler`) in the right order from left to right, as well as a pipeline called `catPipeline` to call the categorical transformers (`indexer` and `encoder`). Then, put these two pipelines `numPipeline` and `catPipeline` into one pipeline.

// COMMAND ----------

import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}

val numPipeline = new Pipeline().setStages(Array(imputer, va, scaler))
val catPipeline = new Pipeline().setStages(Array(indexer, encoder))
val pipeline = new Pipeline().setStages(Array(numPipeline, catPipeline))
val newHousing = pipeline.fit(renamedHousing).transform(renamedHousing)

newHousing.select("scaled_attributes", "category_index_vector").show(5, false)

// COMMAND ----------

// MAGIC %md
// MAGIC Now, use `VectorAssembler` to put all attributes of the final dataset `newHousing` into a big vector, and call the new column `features`.

// COMMAND ----------

val va2 = new VectorAssembler()
    .setInputCols(Array("scaled_attributes", "category_index_vector"))
    .setOutputCol("features")
val dataset = va2.transform(newHousing).select("features", "label")

dataset.show(5, false)

// COMMAND ----------

// MAGIC %md
// MAGIC # 5. Make a model
// MAGIC Here we going to make four different regression models:
// MAGIC * Linear regression model
// MAGIC * Decission tree regression
// MAGIC * Random forest regression
// MAGIC * Gradient-booster forest regression
// MAGIC 
// MAGIC But, before giving the data to train a Machine Learning model, let's first split the data into training dataset (`trainSet`) with 80% of the whole data, and test dataset (`testSet`) with 20% of it.

// COMMAND ----------

val Array(trainSet, testSet) = dataset.randomSplit(Array(0.8, 0.2)) //, seed = 1234L) //seed is optional

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5.1. Linear regression model
// MAGIC Now, train a Linear Regression model using the `LinearRegression` class. Then, print the coefficients and intercept of the model, as well as the summary of the model over the training set by calling the `summary` method.

// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression

// train the model
val lr = new LinearRegression()

val lrModel = lr.fit(trainSet)
val trainingSummary = lrModel.summary

println(s"Coefficients: ${lrModel.coefficients}, Intercept: ${lrModel.intercept}")
println(s"RMSE on train data: ${trainingSummary.rootMeanSquaredError}")

// COMMAND ----------

// MAGIC %md
// MAGIC Now, use `RegressionEvaluator` to measure the root-mean-square-erroe (RMSE) of the model on the test dataset.

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

// make predictions on the test data
val predictions = lrModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5.2. Decision tree regression
// MAGIC Repeat what you have done on Regression Model to build a Decision Tree model. Use the `DecisionTreeRegressor` to make a model and then measure its RMSE on the test dataset.

// COMMAND ----------

import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

val dt = new DecisionTreeRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")

// train the model
val dtModel = dt.fit(trainSet)

// make predictions on the test data
val predictions = dtModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5.3. Random forest regression
// MAGIC Let's try the test error on a Random Forest Model. Youcan use the `RandomForestRegressor` to make a Random Forest model.

// COMMAND ----------

import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

val rf = new RandomForestRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")

// train the model
val rfModel = rf.fit(trainSet)

// make predictions on the test data
val predictions = rfModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5.4. Gradient-boosted tree regression
// MAGIC Fianlly, we want to build a Gradient-boosted Tree Regression model and test the RMSE of the test data. Use the `GBTRegressor` to build the model.

// COMMAND ----------

import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

val gb = new GBTRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setMaxIter(10)

// train the model
val gbModel = gb.fit(trainSet)

// make predictions on the test data
val predictions = gbModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC # 6. Hyperparameter tuning
// MAGIC An important task in Machie Learning is model selection, or using data to find the best model or parameters for a given task. This is also called tuning. Tuning may be done for individual Estimators such as LinearRegression, or for entire Pipelines which include multiple algorithms, featurization, and other steps. Users can tune an entire Pipeline at once, rather than tuning each element in the Pipeline separately. MLlib supports model selection tools, such as `CrossValidator`. These tools require the following items:
// MAGIC * Estimator: algorithm or Pipeline to tune (`setEstimator`)
// MAGIC * Set of ParamMaps: parameters to choose from, sometimes called a "parameter grid" to search over (`setEstimatorParamMaps`)
// MAGIC * Evaluator: metric to measure how well a fitted Model does on held-out test data (`setEvaluator`)
// MAGIC 
// MAGIC `CrossValidator` begins by splitting the dataset into a set of folds, which are used as separate training and test datasets. For example with `k=3` folds, `CrossValidator` will generate 3 (training, test) dataset pairs, each of which uses 2/3 of the data for training and 1/3 for testing. To evaluate a particular `ParamMap`, `CrossValidator` computes the average evaluation metric for the 3 Models produced by fitting the Estimator on the 3 different (training, test) dataset pairs. After identifying the best `ParamMap`, `CrossValidator` finally re-fits the Estimator using the best ParamMap and the entire dataset.
// MAGIC 
// MAGIC Below, use the `CrossValidator` to select the best Random Forest model. To do so, you need to define a grid of parameters. Let's say we want to do the search among the different number of trees (1, 5, and 10), and different tree depth (5, 10, and 15).

// COMMAND ----------

import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidator

val paramGrid = new ParamGridBuilder()
    .addGrid(rf.numTrees, Array(1, 5, 10))
    .addGrid(rf.maxDepth, Array(5, 10, 15))
    .build()

val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")

val cv = new CrossValidator()
    .setEstimator(rf)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(3)
    .setParallelism(5)

val cvModel = cv.fit(trainSet)

val predictions = cvModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC # 7. An End-to-End Classification Test
// MAGIC As the last step, you are given a dataset called `data/ccdefault.csv`. The dataset represents default of credit card clients. It has 30,000 cases and 24 different attributes. More details about the dataset is available at `data/ccdefault.txt`. In this task you should make three models, compare their results and conclude the ideal solution. Here are the suggested steps:
// MAGIC 1. Load the data.
// MAGIC 2. Carry out some exploratory analyses (e.g., how various features and the target variable are distributed).
// MAGIC 3. Train a model to predict the target variable (risk of `default`).
// MAGIC   - Employ three different models (logistic regression, decision tree, and random forest).
// MAGIC   - Compare the models' performances (e.g., AUC).
// MAGIC   - Defend your choice of best model (e.g., what are the strength and weaknesses of each of these models?).
// MAGIC 4. What more would you do with this data? Anything to help you devise a better solution?

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC ### Attribute Information
// MAGIC This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables: 
// MAGIC - ID: ID of each client
// MAGIC - LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit)
// MAGIC - SEX: Gender (1=male, 2=female)
// MAGIC - EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
// MAGIC - MARRIAGE: Marital status (1=married, 2=single, 3=others)
// MAGIC - AGE: Age in years
// MAGIC - PAY_1: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
// MAGIC - PAY_2: Repayment status in August, 2005 (scale same as above)
// MAGIC - PAY_3: Repayment status in July, 2005 (scale same as above)
// MAGIC - PAY_4: Repayment status in June, 2005 (scale same as above)
// MAGIC - PAY_5: Repayment status in May, 2005 (scale same as above)
// MAGIC - PAY_6: Repayment status in April, 2005 (scale same as above)
// MAGIC - BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
// MAGIC - BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
// MAGIC - BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
// MAGIC - BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
// MAGIC - BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
// MAGIC - BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
// MAGIC - PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
// MAGIC - PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
// MAGIC - PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
// MAGIC - PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
// MAGIC - PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
// MAGIC - PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
// MAGIC - default.payment.next.month: Default payment (1=yes, 0=no)

// COMMAND ----------

val rawCreditCardDataset = spark
  .read
  .format("csv")
  .option("sep",",")
  .option("inferSchema", "true")
  .option("header", "true")
  .load("dbfs:/FileStore/shared_uploads/angel@igareta.com/ccdefault.csv")
  .withColumnRenamed("PAY_0", "PAY_1")
  .withColumnRenamed("DEFAULT", "label")

rawCreditCardDataset.show(5)

// COMMAND ----------

for (c <- rawCreditCardDataset.columns.toSeq) {
  rawCreditCardDataset.select(count(when(col(c).isNull, c)).alias(c)).show()
}

// COMMAND ----------

// MAGIC %md
// MAGIC ## 7.1 Dataset Preprocessing

// COMMAND ----------

var creditCardDataset = rawCreditCardDataset
  // Consider unknown education labels the same
  .withColumn("EDUCATION", when(col("EDUCATION") > 3, 0).otherwise(col("EDUCATION")))
  .withColumn("MARRIAGE", when(col("MARRIAGE") === 0, 3).otherwise(col("MARRIAGE")))
  // Discretize age column
//   .withColumn("AGE_CLASS", when(col("AGE") >= 55, 4).when(col("AGE") >= 45, 3).when(col("AGE") >= 35, 2).when(col("AGE") >= 25, 1).when(col("AGE") >= 20, 0))

// For each pay, aggregate negative values (pay duly)
for (i <- 1 to 6) {
  val colName = "PAY_" + i
  creditCardDataset = creditCardDataset.withColumn(colName, when(col(colName) < 0, 0).otherwise(col(colName)))
}
// creditCardDataset = creditCardDataset.withColumn("PAY_MEAN", (col("PAY_1") + col("PAY_2") + col("PAY_3") + col("PAY_4") + col("PAY_5") + col("PAY_6")) / 6)

// For each bill, calculate how much the bill increases or decreases respect to the previous month
// for (i <- 2 to 6) {
//   val currentBillCol = "BILL_AMT" + i
//   val previousBillCol = "BILL_AMT" + (i - 1)
//   creditCardDataset = creditCardDataset.withColumn("BILL_AMT_DIFF" + (i - 1), when(col(previousBillCol) === 0, -2).otherwise((col(currentBillCol) - col(previousBillCol)) / col(previousBillCol))) 
// }

// creditCardDataset.select("LIMIT_BAL", "EDUCATION", "AGE_CLASS", "PAY_MEAN", "BILL_AMT_DIFF1").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## 7.2 Dataset Analysis

// COMMAND ----------

// MAGIC %md
// MAGIC Print summary of amount of given credit and age of the card owner.

// COMMAND ----------

creditCardDataset.describe("LIMIT_BAL", "AGE", "PAY_1").show()

// COMMAND ----------

// MAGIC %md
// MAGIC Group by education and calculate the average limit balance per type

// COMMAND ----------

creditCardDataset.groupBy("EDUCATION").agg(avg("LIMIT_BAL").as("LIMIT_BAL_MEAN")).show()

// COMMAND ----------

creditCardDataset.groupBy("label", "SEX").count().sort(asc("label"), asc("SEX")).show()

// COMMAND ----------

creditCardDataset.groupBy("label", "MARRIAGE").count().sort(asc("label"), asc("MARRIAGE")).show()

// COMMAND ----------

creditCardDataset.groupBy("label", "EDUCATION").count().sort(asc("label"), asc("EDUCATION")).show()

// COMMAND ----------

// MAGIC %md
// MAGIC Studying correlation between people attributes 

// COMMAND ----------

val va = new VectorAssembler().setInputCols(Array("LIMIT_BAL", "AGE", "EDUCATION", "SEX", "MARRIAGE", "label")).setOutputCol("info")
val Row(coeff: Matrix) = Correlation.corr(va.transform(creditCardDataset), "info").head
println(s"The standard correlation coefficient:\n ${coeff}")

// COMMAND ----------

val va = new VectorAssembler().setInputCols(Array("PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "label")).setOutputCol("info")
val Row(coeff: Matrix) = Correlation.corr(va.transform(creditCardDataset), "info").head
println(s"The standard correlation coefficient:\n ${coeff}")

// COMMAND ----------

val va = new VectorAssembler().setInputCols(Array("BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "label")).setOutputCol("info")
val Row(coeff: Matrix) = Correlation.corr(va.transform(creditCardDataset), "info").head
println(s"The standard correlation coefficient:\n ${coeff}")

// COMMAND ----------

val va = new VectorAssembler().setInputCols(Array("PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "label")).setOutputCol("info")
val Row(coeff: Matrix) = Correlation.corr(va.transform(creditCardDataset), "info").head
println(s"The standard correlation coefficient:\n ${coeff}")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 7.3 Dataset Preparation

// COMMAND ----------

val numericColumns = Array("LIMIT_BAL") // "AGE", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
val categoricColumns = Array("EDUCATION", "SEX", "MARRIAGE", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
val categoricColumnsOhe = categoricColumns.map(name => s"${name}_OHE")

// COMMAND ----------

val va = new VectorAssembler()
    .setInputCols(numericColumns)
    .setOutputCol("numerical_attributes")
val scaler = new StandardScaler()
    .setInputCol("numerical_attributes")
    .setOutputCol("scaled_attributes")

// COMMAND ----------

val encoder = new OneHotEncoder()
    .setInputCols(categoricColumns)
    .setOutputCols(categoricColumnsOhe)

// COMMAND ----------

val numPipeline = new Pipeline().setStages(Array(va, scaler))
val catPipeline = new Pipeline().setStages(Array(encoder))
val pipeline = new Pipeline().setStages(Array(numPipeline, catPipeline))
val newCreditCardDataset = pipeline.fit(creditCardDataset).transform(creditCardDataset)

newCreditCardDataset.select("scaled_attributes", (categoricColumnsOhe):_*).show()

// COMMAND ----------

val dataset = new VectorAssembler()
    .setInputCols(Array("scaled_attributes") ++ categoricColumnsOhe)
    .setOutputCol("features")
    .transform(newCreditCardDataset)
    .select("features", "label")

dataset.show(5, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## 7.4 Train models

// COMMAND ----------

val Array(trainSet, testSet) = dataset.randomSplit(Array(0.8, 0.2), seed = 1234L)

// COMMAND ----------

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}  
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

def compute_metrics(predictions: org.apache.spark.sql.DataFrame) {
  val predictionAndLabels = predictions.select($"prediction", $"label").as[(Double, Double)].rdd

  // Instantiate a new metrics objects
  val bMetrics = new BinaryClassificationMetrics(predictionAndLabels)
  val mMetrics = new MulticlassMetrics(predictionAndLabels)
  
  println(s"Area under ROC: ${bMetrics.areaUnderROC}")
  println(s"Accuracy = ${mMetrics.accuracy} ")
  println(s"F-measure = ${mMetrics.fMeasure(1)} ")  
  println("Confusion matrix: ")
  println(mMetrics.confusionMatrix)
}

// COMMAND ----------

// MAGIC %md
// MAGIC ### Logistic Regression

// COMMAND ----------

import org.apache.spark.ml.classification.LogisticRegression

val lr = new LogisticRegression()
  .setMaxIter(10)

val paramGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(0.2, 0.1, 0.0))
  .addGrid(lr.elasticNetParam, Array(0.2, 0.1, 0.0))
  .build()

val evaluator = new BinaryClassificationEvaluator()

val cv = new CrossValidator()
    .setEstimator(lr)
    .setEstimatorParamMaps(paramGrid)
    .setEvaluator(evaluator)
    .setNumFolds(4)

// Fit the model
val lrModel = cv.fit(trainSet)

println("Best model")
print(lrModel.bestModel.extractParamMap())

// Test the model
val predictions = lrModel.transform(testSet)

// Print metrics
compute_metrics(predictions)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Decision Tree

// COMMAND ----------

import org.apache.spark.ml.classification.DecisionTreeClassifier

// Train a DecisionTree model.
val dt = new DecisionTreeClassifier()

val paramGrid = new ParamGridBuilder()
  .addGrid(dt.maxDepth, Array(4, 5, 6, 7, 8, 9, 10))
  .build()

val evaluator = new BinaryClassificationEvaluator()

val cv = new CrossValidator()
    .setEstimator(dt)
    .setEstimatorParamMaps(paramGrid)
    .setEvaluator(evaluator)
    .setNumFolds(4)



// Train model. This also runs the indexers.
val dtModel = cv.fit(trainSet)
println("Best model")
print(dtModel.bestModel.extractParamMap())

val predictions = dtModel.transform(testSet)

// Print metrics
compute_metrics(predictions)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Random Forest

// COMMAND ----------

import org.apache.spark.ml.classification.RandomForestClassifier

val rf = new RandomForestClassifier()

val paramGrid = new ParamGridBuilder()
  .addGrid(rf.maxDepth, Array(10, 15)) // .addGrid(rf.maxDepth, Array(4, 5, 6, 7, 8, 9, 10))
  .addGrid(rf.featureSubsetStrategy, Array("all", "sqrt"))
  .addGrid(rf.numTrees, Array(10, 15, 20)) // .addGrid(rf.numTrees, Array(3, 5, 7))
  .build()

val evaluator = new BinaryClassificationEvaluator()

val cv = new CrossValidator()
    .setEstimator(rf)
    .setEstimatorParamMaps(paramGrid)
    .setEvaluator(evaluator)
    .setNumFolds(4)


val rfModel = cv.fit(trainSet)
// Train model. This also runs the indexers.
println("Best model")
print(rfModel.bestModel.extractParamMap())

val predictions = rfModel.transform(testSet)

// Print metrics
compute_metrics(predictions)

// COMMAND ----------

// MAGIC %md
// MAGIC ## 7.5 Conclusions

// COMMAND ----------

// MAGIC %md
// MAGIC ### Variable Selection
// MAGIC In order to choose the best combination of numerical and categorical variables, we first studied distribution and correlation between the variables in the dataset. Some of the data preprocessing tasks were changing categorical variables that had different values to represent the same meaning (like the values [0, 3, 4, 5] in education or the values [-2, -1, 0] in PAY_X). Once this was done, after finding that the variables starting with BILL_AMTX had a significant correlation among them, we generated new variables BILL_AMT_DIFFX that aimed to represent how much the bills increase or decrease respect to the previous month.
// MAGIC 
// MAGIC ### Dataset Preparation
// MAGIC After the variable selection, we applied a pipeline consisting of a vector assembler and a scaler to standardize the numeric values and also applied one-hot encoding over the categorical variables.
// MAGIC 
// MAGIC ### Algorithms
// MAGIC The algorithms applied to the dataset were:
// MAGIC - Logistic Regression: a Machine Learning method that uses logistic function to model a binary dependent variable.
// MAGIC - Decision Tree: a flowchat-like tree structure, used in classification and regression, where every node represents a test on an attribute and every branch the outcome. 
// MAGIC - Random Forest: an estimator that fits a number of decision tree clasifiers on sub-samples of dataset and uses averaging to improve the accuracy and reduce the overfitting.
// MAGIC 
// MAGIC ### Metrics
// MAGIC The proposed algorithms were analyzed according to the following metrics:
// MAGIC - Confusion matrix: a matrix that is used to describe the performance of a classification model on the test data set and highlighting the quantity of observed True Positives (TP), False Positives (FP), True Negative (TN), and False Negatives (FN).
// MAGIC - Area Under the Curve (AUC):  provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example.
// MAGIC - Accuracy: measures the fraction of predictions the model got right. It is calculated as (TP + TN)/ (TP + TN + FP + FN)
// MAGIC - F1-Score: another measure of test accuracy, which can be obtained by calculating Recall and Precision from the confusion matrix. F1-score = (2*Precision*Recall)/(Precision + Recall)
// MAGIC 
// MAGIC ### Methodology
// MAGIC For each algorithm, we performed hypertuning of parameters using cross validation with 4 folds, comparing the best model attending to the AUC. The results were:
// MAGIC #### Logistic Regression
// MAGIC For the logist regression algorithm, the best combination of parameters (elasticNetParam = 0.0, regParam = 0.1) obtained the following results:
// MAGIC - Area under ROC: 0.6248849574005612
// MAGIC - Accuracy = 0.8169814474343975 
// MAGIC - F-measure = 0.40326975476839244 
// MAGIC 
// MAGIC Confusion matrix: 
// MAGIC 
// MAGIC 4518.0  187.0  
// MAGIC 908.0   370.0 
// MAGIC 
// MAGIC #### Decision Tree
// MAGIC For the decision tree algorithm, the best combination of parameters (maxDepth = 5) obtained the following results:
// MAGIC - Area under ROC: 0.6540963813344111
// MAGIC - Accuracy = 0.8163128865117834 
// MAGIC - F-measure = 0.4631167562286273 
// MAGIC 
// MAGIC Confusion matrix: 
// MAGIC 
// MAGIC 4410.0  295.0  
// MAGIC 804.0   474.0 
// MAGIC 
// MAGIC #### Random Forest
// MAGIC For the random forest algorithm, the best combination of parameters (maxDepth = 5, numTrees = 20) obtained the following results:
// MAGIC - Area under ROC: 0.6530485665201505
// MAGIC - Accuracy = 0.8204913922781213 
// MAGIC - F-measure = 0.4619238476953907 
// MAGIC 
// MAGIC Confusion matrix: 
// MAGIC 
// MAGIC 4448.0  257.0  
// MAGIC 817.0   461.0  
// MAGIC 
// MAGIC ### Conclusion
// MAGIC The best performant model according was the Decision tree, besides the fact that it had lower accuracy, but higher AUC and speed, as outperforms logistic regression and compared to the random forest is only using one tree.
