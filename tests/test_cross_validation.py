"""
Unit Tests for CrossValidation class
"""
import unittest

from pyspark.sql.dataframe import DataFrame as SparkDataFrame
from pyspark.ml.recommendation import ALSModel, ALS
from pandas import DataFrame as PandasDataFrame

from recommend.app.src.recommendation_engine import RecommendationEngine
from recommend.app.src.cross_validation import CrossValidation
from recommend.tests.data_generator import generate_data


class TestCrossValidation(unittest.TestCase):
    # Create engine and load data for testing
    engine = RecommendationEngine()
    test_data = generate_data()
    engine.load_data(test_data.to_json())

    # Train model
    engine.train_model()

    # Create CrossValidation object
    cv = CrossValidation(engine)

    def test_fit(self):
        # ensure the self.best_params of the self.best_model match the model returned
        cv = 2
        scale = True

        parameter1 = "rank"
        values1 = [5, 10]

        parameter2 = "alpha"
        values2 = [0.5, 0.4]

        params = {parameter1: values1, parameter2: values2}
        # Output optimum model from param_grid, stores optimum model and params
        result = self.cv.fit(cv, params, scale)

        self.assertTrue(isinstance(result, ALSModel),
                        "Unexpected datatype returned, expecting ALSModel object.")

        self.assertTrue(self.cv.best_params["rank"] in params["rank"],
                        "Unexpected parameter in best_model, it wasn't part of the test params.")

    def test_create_splits(self):
        # Input: dataframe, n_splits
        data = self.engine.get_data()
        n_splits = 10
        # Return list of dataframe tuples: [(train_1, test_1), (train_2, test_2),...]
        result = self.cv.create_splits(data, n_splits)

        self.assertTrue(len(result) == n_splits, "Unexpected number of splits returned.")
        self.assertTrue(isinstance(result, list),
                        "Unexpected datatype returned, expecting list.")
        self.assertTrue(isinstance(result[0], tuple),
                        "Unexpected datatype returned, expecting tuple.")
        self.assertTrue(isinstance(result[0][0], SparkDataFrame),
                        "Unexpected datatype returned, expecting SparkDataFrame")

    def test_build_models(self):
        # Input dict {metric: parameters}
        params = {"alpha": [.5, .6], "rank": [1, 5]}

        # Since there is only one parameter in this case, otherwise add all together
        total_combinations = 1
        for v in params.values():
            total_combinations *= len(v)

        # build param grid
        # Output list of models for all param combinations
        result, params_result = self.cv.build_models(params)

        self.assertTrue(len(result) == total_combinations,
                        "Unexpected number of models returned.")
        self.assertTrue(isinstance(result, list),
                        "Unexpected datatype returned, expecting list.")
        self.assertTrue(isinstance(result[0], ALS),
                        "Unexpected datatype returned, expecting ALS object.")

        # todo, assert that the params from each model are created as expected
        self.assertTrue(result[0].getRank() in params["rank"],
                        "Model parameters are not being set properly in CV.")

        self.assertTrue(all([k in params.keys() for k in params_result[0].keys()]),
                        "Unexpected param_list returned, it doesn't match what was passed.")

    def test_process_relevance(self):
        # Input: spark dataframe of user_id, prod_id, and rating
        # Where rating is the number of times a report is accessed
        # Output: pandas dataframe with user_id as index and lists in order of relevance of relevant reports

        result = self.cv.process_relevance(self.engine.get_data().sample(False, .10, seed=0))

        self.assertTrue(isinstance(result, PandasDataFrame),
                        "Unexpected datatype returned, expecting Pandas DataFrame.")
        self.assertTrue(result.columns == ["relevant"],
                        "Unexpected columns, only 'relevant' should exist.")

    def test_process_recommendation(self):
        # Input: spark dataframe of user_id, prod_id, and rating
        # Where rating is the number of times a report is accessed
        # Output: pandas dataframe with user_id as index and lists in order of relevance of relevant reports

        result = self.cv.process_recommendation(self.engine.get_model().recommendForAllUsers(5))

        self.assertTrue(isinstance(result, PandasDataFrame),
                        "Unexpected datatype returned, expecting Pandas DataFrame.")
        self.assertTrue(result.columns == ["recommendations"],
                        "Unexpected columns, only 'relevant' should exist.")

    def test_score(self):
        # Calculate model performance score using process_relevance() output and cv_predict() output
        # Return float from 0 to 1 inclusive
        rel_pandas_df = self.cv.process_relevance(self.engine.get_data().sample(False, .10, seed=0))
        rec_pandas_df = self.cv.process_recommendation(self.engine.get_model().recommendForAllUsers(5))

        score = self.cv.score(rec_pandas_df=rec_pandas_df, rel_pandas_df=rel_pandas_df)

        self.assertTrue(isinstance(score, float),
                        "Unexpected datatype returned, expecting float.")
        self.assertTrue(1 >= score >= 0,
                        "Unexpected value, score equals %s while expected to be between 0 and 1" % str(score))


if __name__ == '__main__':
    unittest.main()
