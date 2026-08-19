import os
import re
import json
import numpy as np

# Suppress HuggingFace hub warnings and parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# Optional dependencies for Word & PowerPoint documents
try:
    import docx
except ImportError:
    docx = None

try:
    import pptx
except ImportError:
    pptx = None

# Initialize OpenAI client pointing to local LM Studio or API with timeout
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
    timeout=8.0
)

# Global embedding model cache
_EMBEDDING_MODEL = None

def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL

def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()

def read_docx(docx_path):
    if docx is None:
        raise ImportError("python-docx is not installed.")
    doc = docx.Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                full_text.append(" | ".join(row_text))
    return "\n".join(full_text)

def read_pptx(pptx_path):
    if pptx is None:
        raise ImportError("python-pptx is not installed.")
    prs = pptx.Presentation(pptx_path)
    text_runs = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                text_runs.append(shape.text.strip())
    return "\n".join(text_runs)

def read_plain_text(file_path):
    encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read().strip()
        except (UnicodeDecodeError, Exception):
            continue
    with open(file_path, 'rb') as f:
        return f.read().decode('utf-8', errors='ignore').strip()

def read_any_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return read_pdf(file_path)
    elif ext in ['.docx', '.doc']:
        try:
            return read_docx(file_path)
        except Exception:
            return read_plain_text(file_path)
    elif ext in ['.pptx', '.ppt']:
        try:
            return read_pptx(file_path)
        except Exception:
            return read_plain_text(file_path)
    else:
        return read_plain_text(file_path)

def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def create_faiss_index_from_chunks(chunks):
    model = get_embedding_model()
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype="float32"))
    return index, model

def extract_questions_fast_rules(user_input):
    """
    Ultra-fast rule-based question extractor (0.001 sec execution).
    """
    questions = []
    sentences = re.split(r'(?<=[.?!])\s+|\n+', user_input.strip())
    
    question_starters = (
        'what', 'how', 'why', 'who', 'where', 'when', 'which', 'whose', 'whom',
        'can', 'could', 'would', 'should', 'is', 'are', 'was', 'were', 'do', 'does', 'did',
        'explain', 'describe', 'list', 'detail', 'compare', 'tell', 'summarize'
    )

    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        s_lower = s.lower()
        if s.endswith('?') or any(s_lower.startswith(w) for w in question_starters):
            clean_q = s if s.endswith('?') else s + '?'
            if clean_q not in questions:
                questions.append(clean_q)

    return questions

def extract_questions(user_input):
    """
    Hybrid high-speed question extraction:
    First attempts ultra-fast rule parsing. If multiple questions are detected, returns instantly.
    Otherwise uses LLM for complex prompts.
    """
    if not user_input or not user_input.strip():
        return []

    # Fast path: rule-based check
    rule_qs = extract_questions_fast_rules(user_input)
    if len(rule_qs) >= 2:
        return rule_qs

    # LLM path for nuanced prompts
    try:
        response = client.chat.completions.create(
            model="qwen2.5-3b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract ALL individual questions asked in the user text as a clean JSON array of strings. "
                        "Return ONLY valid JSON array format, e.g. [\"Question 1\", \"Question 2\"]."
                    )
                },
                {"role": "user", "content": user_input}
            ],
            temperature=0.1,
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        
        parsed = json.loads(content)
        if isinstance(parsed, list) and len(parsed) > 0:
            return [str(q).strip() for q in parsed if str(q).strip()]
    except Exception:
        pass

    # Fallback to rule questions or full input
    if rule_qs:
        return rule_qs
    
    clean_input = user_input.strip()
    if not clean_input.endswith('?'):
        clean_input += '?'
    return [clean_input]

def retrieve_context(question, model, index, chunks, k=2):
    if not index or not chunks:
        return ""
    question_embedding = model.encode([question])
    distances, indices = index.search(
        np.array(question_embedding, dtype="float32"),
        min(k, len(chunks))
    )
    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return "\n\n".join(retrieved_chunks)

def answer_single_question(question, context=""):
    if context:
        prompt_content = f"Context:\n{context}\n\nQuestion:\n{question}"
        system_content = "Answer the question strictly and concisely using the provided context. Keep it direct and short."
    else:
        prompt_content = f"Question:\n{question}"
        system_content = "Answer the question concisely."

    try:
        response = client.chat.completions.create(
            model="qwen2.5-3b-instruct",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.3,
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception:
        if context:
            return f"Based on your document context:\n\n{context[:400]}..."
        else:
            return "Unable to connect to LLM server (LM Studio)."

def process_multi_question(user_input, chunks=None, index=None):
    extracted_qs = extract_questions(user_input)
    model = get_embedding_model() if (chunks and index) else None

    results = []
    for i, q in enumerate(extracted_qs, 1):
        context = ""
        if chunks and index and model:
            context = retrieve_context(q, model, index, chunks, k=2)

        answer = answer_single_question(q, context=context)
        results.append({
            "id": i,
            "question": q,
            "answer": answer,
            "context": context if context else None
        })

    return {
        "original_input": user_input,
        "question_count": len(results),
        "results": results
    }
