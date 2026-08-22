from flask import Flask, render_template, request, jsonify
import os
import time
from rag import (
    read_any_document,
    create_chunks,
    create_faiss_index_from_chunks,
    process_multi_question_store,
    get_embedding_model
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'doc-chatbot-secret-key-2026'

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store for all uploaded & indexed documents
# Key: filename -> Value: { filepath, filename, chunks, index, total_chunks, file_size, uploaded_at }
DOCUMENTS_STORE = {}

def get_doc_summary_list():
    summary = []
    for fname, data in DOCUMENTS_STORE.items():
        summary.append({
            "filename": fname,
            "total_chunks": data.get("total_chunks", 0),
            "file_size": data.get("file_size", "Unknown"),
            "uploaded_at": data.get("uploaded_at", "")
        })
    return summary

def format_file_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/status", methods=["GET"])
@app.route("/documents", methods=["GET"])
def get_documents():
    docs = get_doc_summary_list()
    return jsonify({
        "status": "success",
        "has_document": len(docs) > 0,
        "count": len(docs),
        "documents": docs
    })

@app.route("/upload", methods=["POST"])
def upload():
    if "document" not in request.files and "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400

    file = request.files.get("document") or request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"status": "error", "message": "No file selected."}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        text = read_any_document(filepath)
        if not text or not text.strip():
            return jsonify({"status": "error", "message": f"Could not extract text from '{file.filename}' or file is empty."}), 400

        chunks = create_chunks(text)
        index, _ = create_faiss_index_from_chunks(chunks)

        size_bytes = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        file_size_str = format_file_size(size_bytes)
        upload_time_str = time.strftime("%H:%M:%S")

        DOCUMENTS_STORE[file.filename] = {
            "filename": file.filename,
            "filepath": filepath,
            "chunks": chunks,
            "index": index,
            "total_chunks": len(chunks),
            "file_size": file_size_str,
            "uploaded_at": upload_time_str
        }

        return jsonify({
            "status": "success",
            "message": f"Document '{file.filename}' uploaded and indexed successfully ({len(chunks)} chunks).",
            "filename": file.filename,
            "total_chunks": len(chunks),
            "documents": get_doc_summary_list()
        })

    except Exception as e:
        print(f"Error processing document: {e}")
        return jsonify({"status": "error", "message": f"Error indexing document '{file.filename}': {str(e)}"}), 500

@app.route("/documents/<filename>", methods=["DELETE"])
def delete_document(filename):
    if filename in DOCUMENTS_STORE:
        doc_data = DOCUMENTS_STORE.pop(filename)
        # Attempt to delete file from disk if it exists
        try:
            if os.path.exists(doc_data["filepath"]):
                os.remove(doc_data["filepath"])
        except Exception:
            pass
        return jsonify({
            "status": "success",
            "message": f"Document '{filename}' removed.",
            "documents": get_doc_summary_list()
        })
    return jsonify({"status": "error", "message": f"Document '{filename}' not found."}), 404

@app.route("/clear", methods=["POST"])
def clear_all_docs():
    DOCUMENTS_STORE.clear()
    return jsonify({
        "status": "success",
        "message": "All documents cleared.",
        "documents": []
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_input = data.get("message", "").strip()
    selected_doc = data.get("selected_doc")

    if not user_input:
        return jsonify({"status": "error", "message": "Please enter a prompt or question."}), 400

    result = process_multi_question_store(user_input, DOCUMENTS_STORE, selected_doc=selected_doc)

    return jsonify({
        "status": "success",
        "data": result
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)