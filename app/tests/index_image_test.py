import pytest
from flask.testing import FlaskClient
from io import BytesIO

@pytest.mark.order(1)
def test_upload(test_client: FlaskClient, sample_id: str, sample_images: list[tuple[str, bytes]]):
    for i in range(sample_images.__len__()):
        # Attach sample ad
        sample = (BytesIO(sample_images[i][1]), f"ad{i+1}.png")
        
        # Index image
        res = test_client.post('/api/images/index',
                            data={"data": sample, "id": f"{sample_id}_{i}"},
                            content_type="multipart/form-data")

        # Check if succeeded
        assert res.status_code == 200
