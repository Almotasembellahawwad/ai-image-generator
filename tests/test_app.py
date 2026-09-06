import io
import os
import unittest
from unittest.mock import patch

from PIL import Image

from app import app


class FakeClient:
    def __init__(self, **_kwargs):
        pass

    def text_to_image(self, *_args, **_kwargs):
        return Image.new('RGB', (16, 16), '#336699')


class AppTests(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_health_reports_missing_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.get('/api/health')
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json['configured'])

    def test_rejects_invalid_prompt(self):
        response = self.client.post('/generate', json={'prompt': 'x'})
        self.assertEqual(response.status_code, 400)

    def test_generates_png_with_current_client(self):
        with patch.dict(os.environ, {'HF_TOKEN': 'test-token'}), patch('app.InferenceClient', FakeClient):
            response = self.client.post('/generate', json={'prompt': 'editorial portrait', 'style': 'photo', 'aspect': 'portrait'})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json['image'].startswith('data:image/png;base64,'))
            encoded = response.json['image'].split(',', 1)[1]
            self.assertGreater(len(io.BytesIO(__import__('base64').b64decode(encoded)).getvalue()), 10)


if __name__ == '__main__':
    unittest.main()
