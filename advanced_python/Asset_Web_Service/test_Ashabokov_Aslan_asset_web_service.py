from pdb import set_trace
from flask import cli

import requests
from requests import exceptions
from requests.api import request
from werkzeug.wrappers import response
import pytest
from unittest import mock
from argparse import Namespace
import json

from task_Ashabokov_Aslan_asset_web_service import Asset, app
from task_Ashabokov_Aslan_asset_web_service import (
    parse_cbr_currency_base_daily,
    parse_cbr_key_indicators,
    CBR_DAILY_URL,
    CBR_KEY_INDICATORS_URL,
    API_CBR_DAILY,
    API_CBR_KEY_INDICATORS,
    API_503_MESSAGE,
    API_404_MESSAGE,
    API_ASSET_LIST,
    API_GET_ASSET,
    API_CALCULATE_REVENUE,
    API_CLEANUP,
)


# Parser tests

EPS = 10e-8
CBR_DAILY_HTML = "cbr_currency_base_daily.html"
CBR_KEY_INDICATORS_HTML = "cbr_key_indicators.html"
API_UNKNOWN_ROUTE = "abc/gg/wp"

def test_parse_cbr_currency_base_daily():
    '''Test parse_cbr_currency_base_daily'''
    with open(CBR_DAILY_HTML) as fin:
        html_dump = fin.read()

    result_dict = parse_cbr_currency_base_daily(html_dump)
    AUD_rate = result_dict['AUD']
    AMD_rate = result_dict['AMD']

    assert 34 == len(result_dict), (
        f'result_dict should contain 35 items.\nGot: {len(result_dict)}'
    )
    assert abs(AUD_rate - 57.0229) < EPS, (
        f"AUD value should be: 57.0229.\nGot: {AUD_rate}"
    )
    assert abs(AMD_rate - 14.4485 / 100) < EPS, (
        f"AMD value should be: {14.4485 / 100}.\nGot: {AMD_rate}"
    )

def test_parse_parse_cbr_key_indicators():
    '''Test parse_cbr_key_indicators'''
    with open(CBR_KEY_INDICATORS_HTML) as fin:
        html_dump = fin.read()

    result_dict = parse_cbr_key_indicators(html_dump)
    USD_rate = result_dict['USD']
    EUR_rate = result_dict['EUR']
    AU_rate = result_dict['Au']
    AG_rate = result_dict['Ag']
    PT_rate = result_dict['Pt']
    PD_rate = result_dict['Pd']

    for tag in result_dict:
        assert isinstance(result_dict[tag], float), (
            f"Value should type of float.\nGot: {type(result_dict[tag])}"
        )
    assert abs(USD_rate - 75.3498) < EPS, (
        f"USD should be: 75.3498.\nGot: {USD_rate}"
    )
    assert abs(EUR_rate - 92.0699) < EPS, (
        f"EUR should be: 92.0699.\nGot: {EUR_rate}"
    )
    assert abs(AU_rate - 4529.59) < EPS, (
        f"Au should be: 4529.59.\nGot: {AU_rate}"
    )
    assert abs(AG_rate - 62.52) < EPS, (
        f"Ag should be: 62.52.\nGot: {AG_rate}"
    )
    assert abs(PT_rate - 2459.96) < EPS, (
        f"Pt should be: 2459.96.\nGot: {PT_rate}"
    )
    assert abs(PD_rate - 5667.14) < EPS, (
        f"Pd should be: 5667.14.\nGot: {PD_rate}"
    )

# API tests

@pytest.fixture
def client():
    '''Get flask client fixture'''
    with app.test_client() as client:
        yield client

@mock.patch("requests.get")
def test_local_cbr_daily_callback(mock_get, client):
    '''Test local cbr_daily_callback'''
    with open(CBR_DAILY_HTML) as fin:
        html_dump = fin.read()
    response = Namespace()
    response.status_code = 200
    response.text = html_dump
    response.ok = True
    mock_get.return_value = response
    client_response = client.get(API_CBR_DAILY)
    
    assert 200 == client_response.status_code, (
        f"Status code should be 200.\nGot: {client_response.status_code}"
    )
    AMD_rate = json.loads(client_response.data)['AMD']
    assert abs(0.144485 - AMD_rate) < EPS, (
        f"AMD rate should be: 0.144485.\nGot: {AMD_rate}"
    )

def test_cbr_daily_callback(client):
    '''Test cbr_daily_callback with real request'''
    response = client.get(API_CBR_DAILY)

    assert 200 == response.status_code, (
        f"Status code should be 200.\nGot: {response.status_code}"
    )
    assert "application/json" == response.content_type, (
        f"Content type should be application/json.\nGot: {response.content_type}"
    )
    assert response.is_json, (
        f"Response should be json."
    )

@mock.patch("requests.get")
def test_local_cbr_indicators_callback(mock_get, client):
    '''Test local cbr_indicators_callback'''
    with open(CBR_KEY_INDICATORS_HTML) as fin:
        html_dump = fin.read()
    response = Namespace()
    response.status_code = 200
    response.text = html_dump
    response.ok = True
    mock_get.return_value = response
    client_response = client.get(API_CBR_KEY_INDICATORS)

    assert 200 == client_response.status_code, (
        f"Status code should be 200.\nGot: {client_response.status_code}"
    )

def test_cbr_indicators_callback(client):
    '''Test cbr_indicators_callback with real request'''
    response = client.get(API_CBR_KEY_INDICATORS)

    assert 200 == response.status_code, (
        f"Status code should be 200.\nGot: {response.status_code}"
    )
    assert "application/json" == response.content_type, (
        f"Content type should be application/json.\nGot: {response.content_type}"
    )

def test_page_not_found(client):
    '''Test 404 error'''
    response = client.get(API_UNKNOWN_ROUTE)

    # from pdb import set_trace; set_trace();

    assert 404 == response.status_code, (
        f"Status code should be 404.\nGot: {response.status_code}"
    )
    assert not response.is_json, (
        f"Error response should not be json."
    )
    assert API_404_MESSAGE in response.get_data().decode(), (
        f"Text should be: {API_404_MESSAGE}.\nGot: {response.get_data().decode()}"
    )

@mock.patch("requests.get")
def test_page_is_not_accessable(mock_get, client):
    '''Test error 503'''

    def lambda_function(x):
        raise requests.exceptions.ConnectionError()

    TRUE_STATUS_CODE = 503
    mock_get.side_effect = lambda_function

    # from pdb import set_trace; set_trace();

    response = client.get(API_CBR_DAILY)

    assert TRUE_STATUS_CODE == response.status_code, (
        f"Status code should be {TRUE_STATUS_CODE}.\nGot: {response.status_code}"
    )
    assert API_503_MESSAGE == response.get_data().decode(), (
        f"Response message should be: {API_503_MESSAGE}.\nGot: {response.get_data().decode()}"
    )

def create_test_asset_url(name: str) -> str:
    '''Create URL for adding test asset'''
    char_code = 'USD'
    capital = 1
    interest = 0.5
    request_url = f"/api/asset/add/{char_code}/{name}/{capital}/{interest}"

    return request_url

def create_test_asset_get_url(
    name_1: str = None,
    name_2: str = None,
):
    '''Help function for testing get asset callback'''
    name_1 = name_1 or "MyAsset1"
    name_2 = name_2 or "MyAsset2"
    request_url = API_GET_ASSET + f"?name={name_1}&name={name_2}"

    return request_url

def create_test_revenue_request_url(
    period_1: str = None,
    period_2: str = None,
):
    '''Help function for create calculate_revenue request'''
    period_1 = period_1 or 5
    period_2 = period_2 or 10
    request_url = API_CALCULATE_REVENUE + f"?period={period_1}&period={period_2}"

    return request_url

def test_create_asset_callback(client):
    '''Test create_asset_callback'''
    name = "MyAsset"
    request_url = create_test_asset_url(name)
    response = client.get(request_url)

    status_code = 200
    msg = f"Asset {name} was successfully added"

    assert status_code == response.status_code, (
        f"Status code should be: {status_code}.\nGot: {response.status_code}"
    )
    assert not response.is_json, (
        f"Response should be text."
    )
    response_text = response.get_data().decode()
    target_text = msg.format(name)
    assert target_text == response_text, (
        f"Response should be: {response_text}.\nGot: {target_text}"
    )

    status_code = 403
    msg = f"Asset {name} already exists"
    response = client.get(request_url)

    assert status_code == response.status_code, (
        f"Status code should be: {status_code}.\nGot: {response.status_code}"
    )
    assert not response.is_json, (
        f"Response should be text."
    )
    response_text = response.get_data().decode()
    target_text = msg.format(name)
    assert target_text == response_text, (
        f"Response should be: {response_text}.\nGot: {target_text}"
    )

def test_get_asset_list_callback(client):
    '''Test load_asset_from_file'''
    first_asset_name = "MyAsset1"
    second_asset_name = "MyAsset2"
    add_asset_1_url = create_test_asset_url(first_asset_name)
    add_asset_2_url = create_test_asset_url(second_asset_name)
    response = client.get(add_asset_2_url)

    assert 200 == response.status_code, (
        "Status code must be 200."
    )
    response = client.get(add_asset_1_url)
    assert 200 == response.status_code, (
        "Status code must be 200."
    )
    response = client.get(API_ASSET_LIST)

    assert 200 == response.status_code, (
        "Status code must be 200."
    )
    assert response.is_json, (
        "Response must be json."
    )

    res_lst = json.loads(response.get_data())

    assert 3 == len(res_lst), (
        f"Response len must be 2.\nGot: {len(res_lst)}"
    )
    assert first_asset_name == res_lst[1][1], (
        f"Asset {first_asset_name} must be first.\nGot: {res_lst}"
    )

def test_get_asset_callback(client):
    '''Test get_asset_callback'''
    request_url = create_test_asset_get_url()
    response = client.get(request_url)

    # from pdb import set_trace; set_trace();

    assert 200 == response.status_code, (
        "Status code must be 200."
    )

    res_lst = json.loads(response.get_data())

    assert 2 == len(res_lst), (
        f"Response len must be 2.\nGot: {len(res_lst)}"
    )
    assert "MyAsset2" == res_lst[1][1], (
        f"Asset MyAsset2 must be first.\nGot: {res_lst}"
    )

def test_calculate_revenue_callback(client):
    '''Test calculate_revenue_callback'''
    request_url = create_test_revenue_request_url()
    response = client.get(request_url)

    assert 200 == response.status_code, (
        f"Response status code must be 200.\nGot: {response.status_code}"
    )

def test_cleanup_callback(client):
    '''Test cleanup_callback'''
    response = client.get(API_CLEANUP)
    true_msg = "there are no more assets"
    response_msg = response.get_data().decode()

    assert 200 == response.status_code, (
        f"Status code must be 200.\nGot: {response.status_code}"
    )
    assert true_msg == response_msg, (
        f"Msg must be: {true_msg}.\nGot: {response_msg}"
    )

    response = client.get(API_ASSET_LIST)

    assert 200 == response.status_code, (
        f"Status code must be 200.\nGot: {response.status_code}"
    )
    res_lst = json.loads(response.get_data().decode())
    assert 0 == len(res_lst), (
        f"Must return empty list.\nGot: {res_lst}"
    )
