from flask import Flask, request, jsonify, render_template
import fitz  # PyMuPDF for PDF text extraction
import ollama  # Local LLM model

app = Flask(__name__)

def extract_text_from_pdf(pdf_path):
    """Extracts text with page & line numbers from PDF"""
    doc = fitz.open(pdf_path)
    extracted_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        lines = text.split("\n")

        for line_num, line in enumerate(lines):
            extracted_text.append({
                "page": page_num + 1,  # Page numbers start from 1
                "line": line_num + 1,
                "text": line.strip()
            })

    return extracted_text

def get_key_arguments(text):
    """Sends document text to Ollama (Mistral) for legal argument extraction"""
    prompt = f"""
    You are a legal assistant analyzing a legal brief. 
    Extract the **10 most important legal arguments** (both **for and against** the case).
    Include:
    - A **short summary** of each argument.
    - The **exact page & line number** where it appears.

    Document Text:
    {text}

    Output Format:
    - Argument: <summary>
      Page: <page number>
      Line: <line number>
    """

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

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

    pdf_path = f"uploads/{file.filename}"
    file.save(pdf_path)

    extracted_text = extract_text_from_pdf(pdf_path)

    # Prepare text for LLM
    combined_text = "\n".join([f"{item['text']} (Page {item['page']}, Line {item['line']})" for item in extracted_text])

    # Get key legal arguments
    key_arguments = get_key_arguments(combined_text)

    return jsonify({"arguments": key_arguments})

if __name__ == "__main__":
    app.run(debug=True)
