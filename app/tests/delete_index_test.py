import pytest
from flask.testing import FlaskClient

@pytest.mark.order(5)
def test_delete(test_client: FlaskClient, sample_id: str, sample_images: list[tuple[str, bytes]]):
    for i in range(sample_images.__len__()):
        # Delete image index
        res = test_client.delete('/api/images/index', json={"id": f"{sample_id}_{i}"})
        
        # Check if succeeded
        assert res.status_code == 200
