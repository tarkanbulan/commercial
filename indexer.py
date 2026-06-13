#!/usr/bin/env python3
"""T2SAIM RAG Indexer — B: surucusundaki tum belgeleri indexler"""
import os, glob, hashlib
from datetime import datetime

# ChromaDB
import chromadb
from chromadb.utils import embedding_functions

# Haystack
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

# Hedef
DB_DIR = r"B:\T2SAIM_RAG_DB"
TARANACAK = [
    r"B:\T2SAIM_NEXUS\*.md",
    r"B:\T2SAIM_NEXUS\*.txt",
    r"B:\T2SAIM_NEXUS\*.html",
    r"B:\T2SAIM_KRİZ_LAB\*.md",
    r"B:\T2SAIM_KRİZ_LAB\*.txt",
    r"B:\T2SAIM_KRİZ_LAB\*.html",
    r"B:\T2SAIM_James_Projects\*.md",
    r"B:\T2SAIM_James_Projects\*.txt",
    r"B:\T2SAIM_Spock_Hermes\*.md",
    r"B:\T2SAIM_Spock_Hermes\*.txt",
]

os.makedirs(DB_DIR, exist_ok=True)

print("="*50)
print("📚 T2SAIM RAG INDEXER")
print(f"   Baslangic: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*50)

# 1. Belgelari tarama
print("\n1. Belgeler taranıyor...")
docs = []
for pattern in TARANACAK:
    for fp in glob.glob(pattern, recursive=True):
        try:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            doc = {
                'id': hashlib.md5(fp.encode()).hexdigest(),
                'content': content[:10000],  # max 10K karakter
                'meta': {
                    'source': fp,
                    'type': fp.split('.')[-1],
                    'size': len(content)
                }
            }
            docs.append(doc)
        except:
            pass

print(f"   {len(docs)} belge bulundu")

# 2. ChromaDB'ye indexle
print("\n2. ChromaDB indexleniyor...")
client = chromadb.PersistentClient(path=DB_DIR)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Koleksiyon olustur
try:
    client.delete_collection("t2saim_corpus")
except:
    pass

collection = client.create_collection(
    name="t2saim_corpus",
    embedding_function=sentence_transformer_ef
)

# Chunk'lara bol ve ekle
chunk_size = 500
all_chunks = []
all_ids = []
all_metas = []

for doc in docs:
    content = doc['content']
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i+chunk_size]
        if len(chunk.strip()) < 50:
            continue
        chunk_id = f"{doc['id']}_{i}"
        all_chunks.append(chunk)
        all_ids.append(chunk_id)
        all_metas.append(doc['meta'])

# Batch olarak ekle
batch_size = 100
for i in range(0, len(all_chunks), batch_size):
    batch_end = min(i + batch_size, len(all_chunks))
    collection.add(
        documents=all_chunks[i:batch_end],
        ids=all_ids[i:batch_end],
        metadatas=all_metas[i:batch_end]
    )
    if (i // batch_size) % 5 == 0:
        print(f"   {batch_end}/{len(all_chunks)} chunk indexlendi")

print(f"\n✅ Indexleme tamam: {len(all_chunks)} chunk, {len(docs)} belge")
print(f"   Veritabani: {DB_DIR}")

# 3. Test sorgusu
print("\n3. Test sorgusu: '2001 krizi'")
results = collection.query(query_texts=["2001 krizi"], n_results=3)
for i, doc in enumerate(results['documents'][0]):
    src = results['metadatas'][0][i]['source']
    print(f"\n  [{i+1}] Kaynak: {src}")
    print(f"       {doc[:200]}...")

print("\n✅ Sistem hazir! Sorgu ornegi:")
print('   results = collection.query(query_texts=["soru"], n_results=5)')
