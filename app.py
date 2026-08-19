from flask import Flask, render_template, request, jsonify
import os
from rag import read_any_document, create_chunks, create_faiss_index_from_chunks, process_multi_question, get_embedding_model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'doc-chatbot-secret-key-2026'

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CURRENT_DOC = {
    "filename": None,
    "filepath": None,
    "chunks": None,
    "index": None,
    "total_chunks": 0
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/status", methods=["GET"])
def get_status():
    if CURRENT_DOC["filename"]:
        return jsonify({
            "has_document": True,
            "filename": CURRENT_DOC["filename"],
            "total_chunks": CURRENT_DOC["total_chunks"]
        })
    return jsonify({"has_document": False})

@app.route("/upload", methods=["POST"])
def upload():
    if "document" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400

    file = request.files["document"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected."}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        text = read_any_document(filepath)
        if not text or not text.strip():
            return jsonify({"status": "error", "message": "Could not extract text or file is empty."}), 400

        chunks = create_chunks(text)
        index, _ = create_faiss_index_from_chunks(chunks)

        CURRENT_DOC["filename"] = file.filename
        CURRENT_DOC["filepath"] = filepath
        CURRENT_DOC["chunks"] = chunks
        CURRENT_DOC["index"] = index
        CURRENT_DOC["total_chunks"] = len(chunks)

        return jsonify({
            "status": "success",
            "message": f"Document '{file.filename}' indexed successfully ({len(chunks)} chunks).",
            "filename": file.filename,
            "total_chunks": len(chunks)
        })

    except Exception as e:
        print(f"Error processing document: {e}")
        return jsonify({"status": "error", "message": f"Error indexing document: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"status": "error", "message": "Please enter a prompt or question."}), 400

    chunks = CURRENT_DOC.get("chunks")
    index = CURRENT_DOC.get("index")

    result = process_multi_question(user_input, chunks=chunks, index=index)
    result["document_used"] = CURRENT_DOC["filename"]

    return jsonify({
        "status": "success",
        "data": result
    })

@app.route("/clear", methods=["POST"])
def clear_doc():
    CURRENT_DOC["filename"] = None
    CURRENT_DOC["filepath"] = None
    CURRENT_DOC["chunks"] = None
    CURRENT_DOC["index"] = None
    CURRENT_DOC["total_chunks"] = 0
    return jsonify({"status": "success", "message": "Document index cleared."})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)