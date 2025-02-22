from flask import Flask, render_template, request, jsonify, send_file
import os
import aiofiles
from src.helper import get_csv

app = Flask(__name__)

UPLOAD_FOLDER = "static/docs"
OUTPUT_FOLDER = "static/output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
async def upload_file():
    if "pdf_file" not in request.files:
        return jsonify({"msg": "No file uploaded"}), 400

    file = request.files["pdf_file"]
    if file.filename == "":
        return jsonify({"msg": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(file.read())

    return jsonify({"msg": "success", "pdf_filename": file_path})

@app.route("/analyze", methods=["POST"])
async def analyze():
    data = request.form
    pdf_filename = data.get("pdf_filename")
    
    if not pdf_filename:
        return jsonify({"msg": "Invalid file"}), 400

    output_file = get_csv(pdf_filename)
    return jsonify({"output_file": output_file})

@app.route("/download")
def download():
    output_file = os.path.join(OUTPUT_FOLDER, "QA.csv")
    return send_file(output_file, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)