import pytest
from fastapi.testclient import TestClient
from api import app
import os
import io
from PIL import Image
import numpy as np

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_columns(client):
    response = client.get("/columns")
    assert response.status_code == 200
    assert "columns" in response.json()
    assert "product_id" in response.json()["columns"]

def test_categories(client):
    response = client.get("/categories")
    assert response.status_code == 200
    assert "categories" in response.json()

def test_ext_data(client):
    # Testing for category 3 (Pokemon)
    response = client.get("/ext-data?category_id=3")
    assert response.status_code == 200
    assert "ext_data_columns" in response.json()

def test_price(client):
    # We need a valid product_id. Let's try to query for one first if possible, 
    # or use a known one from the database if we have it.
    # Since we can't easily know a valid ID without querying, 
    # we'll check for 404 for a non-existent one too.
    response = client.get("/price?product_id=999999999")
    assert response.status_code == 404

def test_update(client):
    response = client.post("/update")
    assert response.status_code == 200
    assert response.json() == {"status": "Update started"}

def test_scan(client):
    # Create a dummy image
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/scan",
        files={"image": ("test.png", img_byte_arr, "image/png")}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)
