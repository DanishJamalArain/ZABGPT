# src/helper.py

import os
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables instead of hardcoding keys
load_dotenv()

# These should be set in your .env file
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("PINECONE_API_KEY or OPENAI_API_KEY is missing in environment/.env")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def extract_meeting_number(filename: str):
    """
    Extracts meeting number like '1st', '2nd', '3rd', '10th' -> '1', '2', '3', '10'
    """
    match = re.search(r"(\d{1,3})(st|nd|rd|th)?", filename)
    return match.group(1) if match else None


def extract_meeting_date(filename: str):
    """
    Extracts date patterns like '12 Jan 2024' or '12-Jan-2024' from filename.
    Returns 'YYYY-MM-DD' when possible.
    """
    match = re.search(
        r"(\d{1,2})[ -]?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ -]?(20\d{2}|19\d{2})",
        filename,
        re.I,
    )

    if match:
        try:
            day, month, year = match.groups()
            dt = datetime.strptime(f"{day} {month} {year}", "%d %b %Y")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            day, month, year = match.groups()
            return f"{day}-{month}-{year}"

    match_year = re.search(r"(19|20)\d{2}", filename)
    return match_year.group(0) if match_year else None


def extract_meeting_type(folder_name: str, file_name: str):
    """
    Derives meeting type (AC / BASR / DC) from folder or file name.
    This will be used for clean filtering in the app.
    """
    text = f"{folder_name} {file_name}".lower()
    if "basr" in text:
        return "BASR"
    if "dc" in text and "ac" not in text:
        return "DC"
    if "ac" in text:
        return "AC"
    return None


def load_documents_with_metadata(base_folder_path: str):
    """
    Load .md (Markdown) files from folders and return a list of LangChain Document objects
    with rich metadata.
    """
    documents = []

    for root, dirs, files in os.walk(base_folder_path):
        folder_name = os.path.basename(root)

        for file in files:
            # Only accept Markdown files
            if not file.lower().endswith(".md"):
                continue

            full_path = os.path.join(root, file)
            text = ""

            try:
                # Read markdown as plain text
                with open(full_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading {full_path}: {e}")
                continue

            if text.strip():
                stat = os.stat(full_path)
                meeting_type = extract_meeting_type(folder_name, file)

                metadata = {
                    "base_folder": os.path.basename(base_folder_path),
                    "meeting_folder": folder_name,
                    "meeting_file": file,
                    "meeting_number": extract_meeting_number(file),
                    "meeting_date": extract_meeting_date(file),
                    "meeting_type": meeting_type,
                    "file_extension": os.path.splitext(file)[1].lower(),
                    "file_size_bytes": stat.st_size,
                    "created_time_iso": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_time_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "full_path": full_path,
                    "tags": [
                        folder_name.lower(),
                        *([meeting_type.lower()] if meeting_type else []),
                    ],
                }

                documents.append(Document(page_content=text, metadata=metadata))

    return documents


def split_documents(documents):
    """
    Split documents into manageable chunks for embedding and indexing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


def download_hugging_face_embeddings():
    """
    Download and initialize HuggingFace embeddings.
    Model: all-MiniLM-L6-v2
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
