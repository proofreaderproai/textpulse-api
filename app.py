
#updated
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import logging
import nltk
from textstat import flesch_kincaid_grade  # Import textstat
from nltk.tokenize import word_tokenize
from flask_cors import CORS

app = Flask(__name__)
nltk.download('punkt_tab')

# Download required NLTK data
nltk.download('punkt')

# Load models ONCE at server startup
print("Loading Models...")
tokenizer_humarin = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_humarin = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
ai_detector = pipeline("text-classification", model="roberta-base-openai-detector", device=device)
print("Models Loaded")

def paraphrase_humarin(text, num_beams=5, max_length=512):
    input_ids = tokenizer_humarin.encode(f"paraphrase: {text}", return_tensors="pt", max_length=max_length,
                                        truncation=True).to(model_humarin.device)
    outputs = model_humarin.generate(input_ids, num_beams=num_beams, max_length=max_length,
                                    early_stopping=True)
    return tokenizer_humarin.decode(outputs[0], skip_special_tokens=True)

def calculate_ai_score(text):
    result = ai_detector(text)[0]
    return result['score'] if result['label'] == 'LABEL_1' else (1 - result['score'])

def calculate_burstiness(text):
    words = word_tokenize(text.lower())
    if not words:
        return 0
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    frequencies = list(word_counts.values())
    mean_freq = sum(frequencies) / len(frequencies) if frequencies else 0
    squared_diffs = [(f - mean_freq) ** 2 for f in frequencies]
    variance = sum(squared_diffs) / len(frequencies) if frequencies else 0
    std_dev = variance ** 0.5
    return std_dev / mean_freq if mean_freq > 0 else 0

def calculate_perplexity(model, tokenizer, text):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    neg_log_likelihood = outputs.loss
    return torch.exp(neg_log_likelihood).item()


app = Flask(__name__)
CORS(app)

@app.route('/humanize', methods=['POST'])
def humanize_text():
    try:
        data = request.get_json()
        input_text = data.get('text', '')

        if not input_text:
            return jsonify({'error': 'No text provided'}), 400

        # Enforce the 400-token limit
        input_tokens = tokenizer_humarin.tokenize(input_text)
        if len(input_tokens) > 400:
            return jsonify({'error': 'Input text exceeds the maximum limit of 400 tokens.'}), 400

        paraphrased_text = paraphrase_humarin(input_text)

        # Calculate metrics
        ai_score = calculate_ai_score(paraphrased_text)
        fk_score = flesch_kincaid_grade(paraphrased_text)
        burstiness_score = calculate_burstiness(paraphrased_text)
        perplexity_score = calculate_perplexity(model_humarin, tokenizer_humarin, paraphrased_text)

        # Include metrics in the JSON response
        return jsonify({
            'result': paraphrased_text,
            'ai_score': ai_score,
            'fk_score': fk_score,
            'burstiness_score': burstiness_score,
            'perplexity_score': perplexity_score
        }), 200

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': f'An error occurred: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
