import os
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import openai
import re
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# Set your OpenAI API key (bisa juga pakai os.environ)
openai.api_key = os.getenv('OPENAI_API_KEY')


app = Flask(__name__)
CORS(app)

@app.route('/v1/ingredients/recognize', methods=['POST'])
def recognize_ingredients():
    data = request.get_json()
    image_b64 = data.get('image')
    if not image_b64:
        return jsonify({'error': 'No image provided'}), 400
    try:
        # Decode base64 ke PIL Image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{img_b64}"

        # Kirim ke OpenAI Vision API (gpt-4o)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identify all ingredients in this image and return a JSON array (not text, not markdown, not explanation) with fields: name, quantity, expiry (estimate expiry in days for each item). Only output valid JSON array, no explanation, no markdown, no extra text. "},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ],
            max_tokens=512
        )
        answer = response.choices[0].message.content
        print('[INFO] Model output (ingredients):', answer)
        # Ambil JSON dari output model
        match = re.search(r'(\[.*\]|\{.*\})', answer, re.DOTALL)
        json_str = match.group(1) if match else None
        try:
            parsed = json.loads(json_str)
            # Pastikan semua name di ingredients lower case dan quantity hanya angka
            import re as _re
            def clean_quantity(val):
                if isinstance(val, (int, float)):
                    return val
                if isinstance(val, str):
                    # Ambil angka pertama (bisa desimal)
                    m = _re.search(r"[\d.]+", val)
                    if m:
                        try:
                            if '.' in m.group(0):
                                return float(m.group(0))
                            else:
                                return int(m.group(0))
                        except Exception:
                            return None
                return None
            def lower_names_and_clean_qty(ings):
                for ing in ings:
                    if 'name' in ing and isinstance(ing['name'], str):
                        ing['name'] = ing['name'].lower()
                    if 'quantity' in ing:
                        ing['quantity'] = clean_quantity(ing['quantity'])
                return ings
            if isinstance(parsed, dict) and 'ingredients' in parsed:
                parsed['ingredients'] = lower_names_and_clean_qty(parsed['ingredients'])
                return jsonify(parsed), 200
            elif isinstance(parsed, list):
                parsed = lower_names_and_clean_qty(parsed)
                return jsonify({'ingredients': parsed}), 200
        except Exception as parse_exc:
            pass
        return jsonify({'ingredients': [], 'raw': answer, 'json_attempt': json_str}), 200
    except Exception as e:
        import traceback
        print('[ERROR] Exception:', e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Inisialisasi DeepSeek Reasoner client
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

@app.route('/v1/recipes/recommend', methods=['POST'])
def recommend_recipes():
    data = request.get_json()
    ingredients = data.get('ingredients')
    if not ingredients:
        return jsonify({'error': 'No ingredients provided'}), 400
    try:
        prompt = (
            "Berdasarkan daftar bahan berikut (beserta jumlahnya), rekomendasikan 3 resep masakan yang bisa dibuat. "
            "Untuk setiap resep, berikan nama, deskripsi singkat, dan langkah-langkah. "
            "Bahan yang tersedia (dengan jumlah): " + json.dumps(ingredients, ensure_ascii=False) + ". "
            "Jawab dalam format JSON array dengan fields: name, description, steps (array of string), used_ingredients (array of objek name dan quantity yang digunakan untuk resep tsb), missing_ingredients (array of objek name dan quantity yang digunakan untuk resep tsb, namun tidak ada di data bahan) . "
            "Hanya output JSON array, tanpa penjelasan tambahan."
        )
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = deepseek_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            max_tokens=1024
        )
        # Ambil hasil dari response
        content = response.choices[0].message.content
        print('[INFO] Model output (recipes):', content)
        # Ambil JSON array dari output
        match = re.search(r'(\[.*\]|\{.*\})', content, re.DOTALL)
        json_str = match.group(1) if match else None
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return jsonify({'recipes': parsed}), 200
        except Exception:
            pass
        return jsonify({'recipes': [], 'raw': content, 'json_attempt': json_str}), 200
    except Exception as e:
        import traceback
        print('[ERROR] Exception:', e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)