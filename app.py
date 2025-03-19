from flask import Flask, request, jsonify, render_template
import fitz  # PyMuPDF for PDF text extraction
import ollama  # Local LLM model
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import time
import hashlib
import json
import threading

app = Flask(__name__)

# Cache directory for storing processed results
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Uploads directory for storing uploaded PDFs
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Global lock for thread safety
processing_lock = threading.Lock()

def get_cache_path(pdf_path):
    """Generate a unique cache file path based on the PDF file's hash."""
    with open(pdf_path, 'rb') as f:
        content = f.read(1024 * 1024)  # Read first 1MB for hashing
        file_hash = hashlib.md5(content).hexdigest()
    return os.path.join(CACHE_DIR, f"{file_hash}.json")

def check_cache(pdf_path):
    """Check if we have cached results for this PDF."""
    cache_path = get_cache_path(pdf_path)
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None

def save_to_cache(pdf_path, data):
    """Save results to cache."""
    cache_path = get_cache_path(pdf_path)
    with open(cache_path, 'w') as f:
        json.dump(data, f)

def extract_text_from_pdf(pdf_path):
    """Extract text from strategic pages of the PDF with correct page numbering."""
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    
    if page_count == 0:
        return []

    extracted_text = []
    sample_pages = []

    # Select strategic pages
    sample_pages.append(0)  # First page
    if page_count > 3:
        sample_pages.extend([1, 2])
    if page_count > 10:
        middle = page_count // 2
        sample_pages.extend([middle - 1, middle, middle + 1])
    if page_count >= 5:
        sample_pages.extend([page_count - 2, page_count - 1])

    sample_pages = sorted(set(sample_pages))  # Remove duplicates

    # Parallel text extraction
    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, 4)) as executor:
        futures = {executor.submit(extract_page_text, doc, page_num): page_num for page_num in sample_pages}
        for future in as_completed(futures):
            extracted_text.extend(future.result())

    # Sort by page number and relevance score
    extracted_text.sort(key=lambda x: (x['page'], x['relevance_score']), reverse=True)
    return extracted_text[:20]  # Keep top 20 relevant sections

def extract_page_text(doc, page_num):
    """Extract and rank text from a page with correct page number indexing."""
    page = doc[page_num]
    text = page.get_text("text")
    
    if not text.strip():
        return []  # Skip empty pages

    paragraphs = re.split(r'\n\s*\n', text)
    results = []

    for para_num, para in enumerate(paragraphs):
        para = para.strip().replace('\n', ' ')
        if len(para) < 30:
            continue
        score = calculate_relevance_score(para)
        if score > 0:
            results.append({
                "page": page_num + 1,  # ✅ Convert 0-based index to 1-based
                "text": para, 
                "relevance_score": score
            })

    results.sort(key=lambda x: x['relevance_score'], reverse=True)
    return results[:3]  # Keep top 3 most relevant paragraphs

def calculate_relevance_score(text):
    """Calculate a relevance score based on legal terminology."""
    text_lower = text.lower()
    positive_keywords = ["argue", "contend", "claim", "assert", "plaintiff", "defendant",
                         "evidence", "court", "judgment", "statute", "therefore", "conclude"]
    strong_keywords = ["precedent", "constitutional", "violation", "protected", "supreme court", 
                       "fundamental", "rights", "law", "legal", "justice"]
    
    score = sum(1 for keyword in positive_keywords if keyword in text_lower) 
    score += sum(2 for keyword in strong_keywords if keyword in text_lower)

    if 100 <= len(text) <= 300:
        score += 1  # Bonus for optimal length
    return score

def process_text_with_ollama(extracted_data):
    """Generate legal arguments using Ollama LLM."""
    prompt_text = "Legal Document Analysis:\n\n"
    for item in extracted_data:
        prompt_text += f"Page {item['page']}: {item['text'][:150]}...\n\n"

    prompt = f"""
    Identify 5 arguments FOR and 5 arguments AGAINST based on the legal document excerpts.
    
    FORMAT:
    FOR:
    1. [Argument] (Page X)
    2. [Argument] (Page Y)
    
    AGAINST:
    1. [Argument] (Page Z)
    2. [Argument] (Page W)
    
    Text excerpts:
    {prompt_text}
    """

    try:
        response = ollama.chat(
            model="gemma",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,
                "num_predict": 512,
                "top_k": 1,
                "top_p": 0.1,
                "repeat_penalty": 1.2,
                "stop": ["```"]
            },
            stream=False
        )

        content = response.get("message", {}).get("content", "") if isinstance(response, dict) else str(response)
        return parse_ollama_response(content)
    except Exception as e:
        print(f"Error with Ollama: {e}")
        return extract_arguments_rule_based(extracted_data)

def parse_ollama_response(content):
    """Parse LLM response into structured arguments."""
    arguments = {"for": [], "against": []}

    for_section = re.search(r"FOR:?(.*?)(?=AGAINST:|$)", content, re.DOTALL | re.IGNORECASE)
    against_section = re.search(r"AGAINST:?(.*?)(?=$)", content, re.DOTALL | re.IGNORECASE)

    arguments["for"] = extract_numbered_items(for_section.group(1)) if for_section else []
    arguments["against"] = extract_numbered_items(against_section.group(1)) if against_section else []
    return arguments

def extract_numbered_items(text):
    """Extract numbered list items from text."""
    if not text:
        return []
    pattern = r'(?:^|\n)(?:\d+\.\s*|\*\s*|-\s*)(.+?)(?=(?:\n(?:\d+\.\s*|\*\s*|-\s*)|\Z))'
    return [match.strip() for match in re.findall(pattern, text, re.DOTALL)] or text.strip().split('\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    pdf_path = os.path.join(UPLOADS_DIR, file.filename)
    file.save(pdf_path)

    cached_result = check_cache(pdf_path)
    if cached_result:
        return jsonify(cached_result)

    with processing_lock:
        extracted_text = extract_text_from_pdf(pdf_path)
        arguments = process_text_with_ollama(extracted_text)
        save_to_cache(pdf_path, arguments)
    
    return jsonify(arguments)

if __name__ == "__main__":
    app.run(debug=True)

