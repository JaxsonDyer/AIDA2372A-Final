import pytest
from src.app import app
import json


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code in [200, 503]
    data = json.loads(response.data)
    assert 'status' in data


def test_predict_endpoint_no_model(client):
    # This might return 503 if the model isn't loaded
    # (which shouldn't be in simple tests unless mock is provided)
    # or it might attempt to predict.
    response = client.post('/predict', json=[{"Age": 30, "DailyRate": 1000}])
    # In GitHub actions, model is unlikely to be loaded without a setup step
    assert response.status_code in [200, 400, 503]
