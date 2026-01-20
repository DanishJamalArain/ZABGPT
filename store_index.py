# store_index.py

import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import (
    load_documents_with_metadata,
    split_documents,
    download_hugging_face_embeddings,
)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment/.env")

# ✅ BASE PATH - Parent directory containing all 3 folders
BASE_DATA_PATH = r"C:\Users\babad\OneDrive\Desktop\New Approach 2\MyFinal Year Project\ZABGPT\Data"

# All 3 meeting folders
MEETING_FOLDERS = [
    os.path.join(BASE_DATA_PATH, "All AC MoM"),
    os.path.join(BASE_DATA_PATH, "All BASR MoM"),
    os.path.join(BASE_DATA_PATH, "All DC MoM"),
]

print("Loading documents from all meeting folders...")
all_docs = []

for folder_path in MEETING_FOLDERS:
    folder_name = os.path.basename(folder_path)
    print(f"\n📂 Processing: {folder_name}")
    
    if not os.path.exists(folder_path):
        print(f"⚠️  Folder not found: {folder_path}")
        continue
    
    docs = load_documents_with_metadata(folder_path)
    print(f"✅ Loaded {len(docs)} documents from {folder_name}")
    all_docs.extend(docs)

print(f"\n📊 Total documents loaded: {len(all_docs)}")

# Show a few samples to verify metadata
print("\n📋 Sample metadata:")
for d in all_docs[:5]:
    print(f"  - File: {d.metadata.get('meeting_file', 'N/A')}")
    print(f"    Type: {d.metadata.get('meeting_type', 'N/A')}")
    print(f"    Date: {d.metadata.get('meeting_date', 'N/A')}\n")

print("Splitting documents into chunks...")
chunks = split_documents(all_docs)
print(f"Total chunks: {len(chunks)}")

print("\nDownloading embeddings...")
embeddings = download_hugging_face_embeddings()

print("Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "newapproach2"

# Create index if not exists
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"Index '{index_name}' already exists. New data will be upserted.")

print("\nUploading chunks to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("✅ Indexing completed successfully!")
print(f"📈 Total vectors uploaded: {len(chunks)}")