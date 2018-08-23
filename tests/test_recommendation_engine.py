"""
Unit tests for RecommendationEngine

TestCases are split into three parts that correlate with the RecommendationEngine
TestRecommendationEngine...
* Data Methods -> ...Data class
* Model Methods -> ...Model class
* Prediction Methods -> ...Prediction class

Run units test on command line from unit_tests directory with below example command.
spark-submit --master local[2] test_engine.py

"""
import os
import unittest
import glob
from pandas import testing as pandas_testing
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.dataframe import DataFrame as SparkDataFrame
from recommend.tests.data_generator import generate_data
from recommend.app.src.recommendation_engine import RecommendationEngine, InputDataError, DataSchemas as ds
import shutil

# Setup for all tests
engine = RecommendationEngine()
test_data = generate_data()


class TestRecommendationEngineData(unittest.TestCase):
    def test_load_data(self):
        global test_data
        # Convert from dataframe into dict
        data_in = test_data.to_json()
        data_out = engine.load_data(data_in)  # Returns spark dataframe
        # Sort, order columns, reset index for data_out. Focused on content comparison and type only
        df_out = data_out.toPandas().sort_values(by=list(test_data.columns))[test_data.columns].reset_index(drop=True)
        # Test to ensure dataframes are equivalent
        pandas_testing.assert_frame_equal(test_data, df_out)
        self.assertTrue(isinstance(data_out, SparkDataFrame))
        # TODO Assert the column order is user_id, prod_id, rating
        self.assertTrue(all([a == b for a, b in zip(["user_id", "prod_id", "rating"], data_out.schema.names)]),
                        "Columns are not ordered correctly.")
        return

    def test_transform_data(self):

        # todo test for assertion errors when violating the checks within _convert_dtypes
        def check_data_assertion(data, error_type, msg=None):
            err = InputDataError()
            try:
                engine.transform_data(data)
            except InputDataError as error_message:
                err = error_message
            finally:
                if not err[error_type]:
                    raise AssertionError(msg)

        # test df missing columns user_id, prod_id, rating
        local_test_data = engine.sqlContext.createDataFrame(
            [(555, 5555, 1), (444, 4444, 0), (333, 3333, 1)], ("test", "prod_id", "rating"))
        check_data_assertion(local_test_data, "column_error",
                             "Missing 'user_id' column didn't produce InputDataError.")

        # test df that contains characters
        # Test user_id
        local_test_data = engine.sqlContext.createDataFrame(
            [("5t5", 5555, 1), (444, 4444, 0), (333, 3333, 1)], schema=ds.df_schema_str)
        check_data_assertion(local_test_data, "type_error",
                             "Characters in user_id column are not properly resulting in an InputDataError.")

        # Test prod_id
        local_test_data = engine.sqlContext.createDataFrame(
            [(555, "5t55", 1), (444, 4444, 0), (333, 3333, 1)], schema=ds.df_schema_str)
        check_data_assertion(local_test_data, "type_error",
                             "Characters in prod_id column are not properly resulting in InputDataError.")

        # Test rating
        local_test_data = engine.sqlContext.createDataFrame(
            [(555, 5555, "A"), (444, 4444, 0), (333, 3333, 1)], schema=ds.df_schema_str)
        check_data_assertion(local_test_data, "type_error",
                             "Characters in rating column are not properly resulting in InputDataError.")

        # test df that contains duplicate user_id, prod_id pairs
        local_test_data = engine.sqlContext.createDataFrame(
            [(555, 5555, 1), (555, 5555, 1), (444, 4444, 0), (333, 3333, 1)], schema=ds.df_schema_int)
        check_data_assertion(local_test_data, "unique_error",
                             "Duplicate user_id/prod_id pairs are not properly resulting in InputDataError.")

        # test to ensure dtypes match [int, int, not string]
        local_test_data = engine.sqlContext.createDataFrame(
            [(555, 5555, 1), (444, 4444, 0), (333, 3333, 1)], schema=ds.df_schema_str)
        check_data_assertion(local_test_data, "type_error",
                             "String data type rating did not properly result in InputDataError.")

        # test to ensure that no assertion errors occur when data meets specs
        local_test_data = engine.sqlContext.createDataFrame(
            [(555, 5555, 1), (444, 4444, 0), (333, 3333, 1)], schema=ds.df_schema_int)
        try:
            engine.transform_data(local_test_data)
        except InputDataError:
            raise AssertionError("Unexpected InputDataError.")

        # test to ensure that scaled=True performs correctly
        # Create test dataframe
        local_test_data = engine.sqlContext.createDataFrame(
            [(555, 5555, 1), (555, 6666, 2), (444, 4444, 50), (333, 3333, 0)], schema=ds.df_schema_int)
        local_test_data = engine.transform_data(local_test_data, scale=True)  # Transform dataframe using transform_data
        # Create validation dataframe
        validation_data = engine.sqlContext.createDataFrame(
            [(555, 5555, 0), (555, 6666, 100), (444, 4444, 100), (333, 3333, 100)], schema=ds.df_schema_int)

        local_test_data = local_test_data.toPandas()
        validation_data = validation_data.toPandas()
        pandas_testing.assert_frame_equal(local_test_data, validation_data)  # Assert dataframes are equal
        return


class TestRecommendationEngineModel(unittest.TestCase):

    # model = None
    model_path = "test_model"

    def test_load_model(self):
        global engine
        # todo load the persisted model
        # ensure model of class ALS is stored in engine.model
        model = engine.load_model("test_model")
        self.assertTrue(isinstance(model, ALSModel))
        if os.path.exists("test_model"):
            shutil.rmtree("test_model")
        return

    def test_a_persist_model(self):
        global engine
        engine.load_data(test_data.to_json())
        engine.train_model(scale=True)
        self.assertTrue(engine.persist_model(self.model_path))

        # Check folder for contents
        all_files = glob.glob(self.model_path + '/**/*', recursive=True)
        contains_item_factors = os.path.join(self.model_path, "itemFactors") in all_files
        contains_user_factors = os.path.join(self.model_path, "userFactors") in all_files
        contains_metadata = os.path.join(self.model_path, "metadata") in all_files
        self.assertTrue(contains_item_factors, "contains_item_factors is false")
        self.assertTrue(contains_user_factors, "contains_user_factors is false")
        self.assertTrue(contains_metadata, "contains_metadata is false")
        return

    def test_train_model(self):
        global engine
        # todo try running second with explicit=True
        # todo ensure the model performs similar to stored results (using seeds?)
        # ensure MatrixFactorizationModel is returned
        engine.load_data(test_data.to_json())
        model = engine.train_model(scale=True)
        self.assertTrue(isinstance(model, ALSModel))
        return


class TestRecommendationEnginePrediction(unittest.TestCase):
    num_items = 10

    def setUp(self):
        global engine
        engine.load_data(test_data.to_json())
        engine.train_model()
        return

    def test_generate_recommendations(self):
        global engine
        result = engine.generate_recommendations(self.num_items)

        # todo: test script: ensure that the number of recommendations is correct or less than for x number
        lengths_match = [len(result[k]) <= self.num_items for k in list(result.keys())[2:7]]
        self.assertTrue(all(lengths_match), "Result returning an unexpected number of items per users.")
        # todo ensure the returned object is a dictionary
        self.assertTrue(isinstance(result, dict))
        return

    def test_recommend_items(self):
        # Grab a user id from dataframe
        # Ensure the proper number of items are returned and the object is a list
        user_id = engine.get_data().select("user_id").take(1)[0].user_id
        result = engine.recommend_items(user_id=user_id, num_items=self.num_items)

        length_correct = len(result) <= self.num_items

        self.assertTrue(length_correct, "Result returning an unexpected number of items.")
        self.assertTrue(isinstance(result, list))
        return


if __name__ == '__main__':
    unittest.main()
