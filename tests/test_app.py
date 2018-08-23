
# Integration tests for recommend/app.py
# http://flask.pocoo.org/docs/1.0/testing/
# TODO Add to requirements: pytest==3.6.4
import os
import shutil
import json

import pytest

from recommend.app.app import app
from recommend.tests.data_generator import generate_data


@pytest.fixture
def client():
    app.config['TESTING'] = True
    client = app.test_client()

    return client

# TODO Functionality to test:
# Train, recommend, train, recommend has same output
# Train, recommend, save-load, recommend and it still has same output
# Tune, recommend, train, recommend has same output
# Tune, recommend, save, load, recommend has same output

# Test each endpoint individually


def test_scenario_0(client):
    """
    Train, recommend, train, recommend. Assert outputs are equal.
    :param client:
    :return:
    """
    # Set variables
    users = 10
    products = 50
    recommended_items = 20
    # Create artificial data
    data_in = generate_data(unique_user_n=users, unique_prod_n=products).to_json()
    # /train 1
    rv_train_1 = client.post('/train', data={'data': data_in}).data
    # /recommend 1
    rv_recommend_1 = client.post('/recommend', data={"items": recommended_items}).data.decode("utf-8")
    recommend_1 = dict(sorted({int(k): v for k, v in json.loads(rv_recommend_1).items()}.items()))

    # result = requests.post(app_url + '/recommend', data={"items": 20}).content.decode(
    #     "utf-8")  # Generate recommendations
    # print(dict(sorted({int(k): v for k, v in json.loads(result).items()}.items())))
    # /train 2
    rv_train_2 = client.post('/train', data={'data': data_in}).data
    # /recommend 2
    rv_recommend_2 = client.post('/recommend', data={"items": recommended_items}).data.decode("utf-8")
    recommend_2 = dict(sorted({int(k): v for k, v in json.loads(rv_recommend_2).items()}.items()))

    # Assertion statements
    assert rv_train_1.decode("utf-8") == "success", "Error in /train 1"
    assert rv_train_2.decode("utf-8") == "success", "Error in /train 2"
    assert max([len(v) for v in recommend_1.values()]) == recommended_items, "Unexpected number of items recommended 1"
    assert max([len(v) for v in recommend_2.values()]) == recommended_items, "Unexpected number of items recommended 2"
    assert recommend_1 == recommend_2, "Inconsistent results."

    return None


def test_scenario_1(client):
    """
    Train, recommend, save, load, recommend. Assert outputs are equal.
    :param client:
    :return:
    """
    # Set variables
    users = 10
    products = 50
    recommended_items = 20
    model_name = "test_model"
    # Ensure model doesn't already exist or delete
    if os.path.exists(model_name):
        shutil.rmtree(model_name)
    # Create artificial data
    data_in = generate_data(unique_user_n=users, unique_prod_n=products).to_json()
    # /train
    rv_train = client.post('/train', data={'data': data_in}).data
    # /recommend 1
    rv_recommend_1 = client.post('/recommend', data={"items": recommended_items}).data.decode("utf-8")
    recommend_1 = dict(sorted({int(k): v for k, v in json.loads(rv_recommend_1).items()}.items()))
    # /save
    rv_save = client.post('/save', data={"name": model_name}).data
    # /load
    rv_load = client.post('/load', data={"name": model_name}).data
    # /recommend 2
    rv_recommend_2 = client.post('/recommend', data={"items": recommended_items}).data.decode("utf-8")
    recommend_2 = dict(sorted({int(k): v for k, v in json.loads(rv_recommend_2).items()}.items()))

    # Assertion statements
    assert rv_train.decode("utf-8") == "success", "Error in /train."
    assert rv_save.decode("utf-8") == "success", "Error in /save."
    assert rv_load.decode("utf-8") == "success", "Error in /load."
    assert max([len(v) for v in recommend_1.values()]) == recommended_items, "Unexpected number of items recommended. 1"
    assert max([len(v) for v in recommend_2.values()]) == recommended_items, "Unexpected number of items recommended. 2"
    assert recommend_1 == recommend_2, "Inconsistent results."

    # Clean up
    if os.path.exists(model_name):
        shutil.rmtree(model_name)

    return None


def test_scenario_2(client):
    """
    Tune, recommend, train, recommend. Assert outputs are equal.
    :param client:
    :return:
    """
    # Set variables
    users = 10
    products = 50
    recommended_items = 20
    # Create artificial data
    data_in = generate_data(unique_user_n=users, unique_prod_n=products).to_json()
    # /tune
    rv_tune = client.post('/tune', data={'data': data_in}).data
    # /recommend 1
    rv_recommend_1 = client.post('/recommend', data={"items": recommended_items}).data.decode("utf-8")
    recommend_1 = dict(sorted({int(k): v for k, v in json.loads(rv_recommend_1).items()}.items()))
    # /train
    rv_train = client.post('/train', data={'data': data_in}).data
    # /recommend 2
    rv_recommend_2 = client.post('/recommend', data={"items": recommended_items}).data.decode("utf-8")
    recommend_2 = dict(sorted({int(k): v for k, v in json.loads(rv_recommend_2).items()}.items()))

    # Assertion statements
    assert rv_train.decode("utf-8") == "success", "Error in /train."
    assert max([len(v) for v in recommend_1.values()]) == recommended_items, "Unexpected number of items recommended. 1"
    assert max([len(v) for v in recommend_2.values()]) == recommended_items, "Unexpected number of items recommended. 2"
    assert recommend_1 == recommend_2, "Inconsistent results."

    return None


def test_scenario_3(client):
    """
    Tune, recommend, save, load, recommend. Assert outputs are equal.
    :param client:
    :return:
    """
    # Set variables
    users = 10
    products = 50
    recommended_items = 20
    model_name = "test_model"
    # Ensure model doesn't already exist or delete
    if os.path.exists(model_name):
        shutil.rmtree(model_name)
    # Create artificial data
    data_in = generate_data(unique_user_n=users, unique_prod_n=products).to_json()
    # /tune
    rv_tune = client.post('/tune', data={'data': data_in}).data
    # /recommend 1
    rv_recommend_1 = client.post('/recommend', data={"items": recommended_items}).data.decode("utf-8")
    recommend_1 = dict(sorted({int(k): v for k, v in json.loads(rv_recommend_1).items()}.items()))
    # /save
    rv_save = client.post('/save', data={"name": model_name}).data
    # /load
    rv_load = client.post('/load', data={"name": model_name}).data
    # /recommend 2
    rv_recommend_2 = client.post('/recommend', data={"items": recommended_items}).data.decode("utf-8")
    recommend_2 = dict(sorted({int(k): v for k, v in json.loads(rv_recommend_2).items()}.items()))

    # Assertion statements
    assert rv_save.decode("utf-8") == "success", "Error in /save."
    assert rv_load.decode("utf-8") == "success", "Error in /load."
    assert max([len(v) for v in recommend_1.values()]) == recommended_items, "Unexpected number of items recommended. 1"
    assert max([len(v) for v in recommend_2.values()]) == recommended_items, "Unexpected number of items recommended. 2"
    assert recommend_1 == recommend_2, "Inconsistent results."

    # Clean up
    if os.path.exists(model_name):
        shutil.rmtree(model_name)
    return None
