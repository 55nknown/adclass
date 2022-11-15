import os
import re
import uuid
import pytest
from flask.testing import FlaskClient
from app import router


@pytest.fixture(scope='module')
def test_client() -> FlaskClient:
    '''Flask test client to test the API'''
    with router.test_client() as client:
        with router.app_context():
            yield client

@pytest.fixture(scope='session')
def sample_images() -> list[tuple[str, bytes]]:
    '''Loads sample ad images to upload and index'''
    sample_data = []
    samples = "../samples/ads/"
    for sample_path in os.listdir(samples):
        # Only allow .png files
        if not sample_path.endswith(".png"):
            continue
        sample_file = open(f"{samples}{sample_path}", 'rb')
        sample_data.append((sample_path, sample_file.read()))
        sample_file.close()
    # Order sample files by their numbers
    sample_data.sort(key=lambda f: [int(c) for c in re.findall(r'\d+', f[0])][0])
    return sample_data

@pytest.fixture(scope='session')
def sample_id() -> str:
    '''Generates a persistent UUID for indexes'''
    return uuid.uuid4().hex