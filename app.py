import base64
import io
import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from huggingface_hub import InferenceClient

load_dotenv()
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'black-forest-labs/FLUX.1-schnell'
NEGATIVE_PROMPT = 'blurry, low quality, distorted, deformed, watermark, signature, extra limbs, unreadable text'
STYLES = {
    'none': '',
    'photo': 'professional editorial photography, natural light, detailed, realistic',
    'cinematic': 'cinematic composition, dramatic lighting, rich color grading, detailed',
    'illustration': 'polished digital illustration, expressive shapes, detailed composition',
    'minimal': 'minimal composition, clean forms, restrained color palette, generous negative space',
}
SIZES = {
    'square': (1024, 1024),
    'landscape': (1216, 832),
    'portrait': (832, 1216),
}


def hf_token():
    """Prefer the current token name while keeping old deployments compatible."""
    return os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_READ_TOKEN') or os.getenv('HUGGINGFACE_WRITE_TOKEN')


def model_name():
    return os.getenv('HF_IMAGE_MODEL', DEFAULT_MODEL).strip()


def provider_name():
    return os.getenv('HF_PROVIDER', 'auto').strip()


@app.get('/')
def home():
    return render_template('index.html', model=model_name())


@app.get('/api/health')
def health():
    return jsonify({
        'status': 'ready' if hf_token() else 'configuration-required',
        'configured': bool(hf_token()),
        'model': model_name(),
        'provider': provider_name(),
    })


@app.post('/generate')
def generate_image():
    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get('prompt') or '').strip()
    style = str(payload.get('style') or 'none').lower()
    aspect = str(payload.get('aspect') or 'square').lower()

    if len(prompt) < 3:
        return jsonify({'error': 'Describe the image in at least 3 characters.'}), 400
    if len(prompt) > 1_500:
        return jsonify({'error': 'Prompt is too long. Keep it under 1,500 characters.'}), 413
    if style not in STYLES or aspect not in SIZES:
        return jsonify({'error': 'Unsupported style or aspect ratio.'}), 400
    token = hf_token()
    if not token:
        return jsonify({'error': 'Image generation is not configured. Add HF_TOKEN to the deployment environment.'}), 503

    width, height = SIZES[aspect]
    final_prompt = ', '.join(part for part in [prompt, STYLES[style]] if part)
    model = model_name()

    try:
        client = InferenceClient(provider=provider_name(), api_key=token, timeout=120)
        image = client.text_to_image(
            final_prompt,
            model=model,
            negative_prompt=NEGATIVE_PROMPT,
            width=width,
            height=height,
        )
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{encoded}',
            'model': model,
            'aspect': aspect,
        })
    except Exception as error:
        logger.exception('Image generation failed with model %s', model)
        message = str(error).lower()
        if '401' in message or 'unauthorized' in message or 'authentication' in message:
            public_message = 'The Hugging Face token is invalid or expired. Update HF_TOKEN.'
            status = 503
        elif '402' in message or 'credits' in message or 'payment' in message:
            public_message = 'The inference provider has no remaining credits. Add credits or change HF_PROVIDER.'
            status = 503
        elif '403' in message or 'gated' in message:
            public_message = 'This token cannot access the selected model. Accept its terms or change HF_IMAGE_MODEL.'
            status = 503
        elif 'timeout' in message or 'timed out' in message:
            public_message = 'The image provider timed out. Please try again.'
            status = 504
        else:
            public_message = 'The image provider is temporarily unavailable. Please try again shortly.'
            status = 502
        return jsonify({'error': public_message}), status


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({'error': 'Request is too large.'}), 413


if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_ENV') == 'development')
