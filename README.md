# AskDoc - AI Powered PDF Question-Answer Generator  

AskDoc is an intelligent AI-powered educational assistant designed to analyze PDFs, generate insightful questions, and provide precise answers using LLM-powered RAG techniques.
This project leverages Gemini-1.5-Pro for question generation and FAISS-powered vector search for accurate answer retrieval, ensuring that learners can extract key knowledge from documents effortlessly.  

  
**Note**: Gemini-1.5-Pro is limited to only 5 api requests per minute and 50 api requests per day.

## 📂 Project Structure  
📂 AskDoc  
 ┣ 📂 src  
 ┃ ┣ 📜 helper.py  
 ┃ ┣ 📜 prompt.py  
 ┣ 📂 static  
 ┃ ┣ 📂 docs            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     # Uploaded PDFs  
 ┃ ┣ 📂 output          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     # Generated CSV files  
 ┣ 📂 templates  
 ┃ ┣ 📜 index.html  
 ┣ 📜 app.py  
 ┣ 📜 requirements.txt  
 ┣ 📜 README.md  

## ✨ Features
✅ **PDF Processing** – Extracts and splits text from PDFs  
✅ **AI-Powered Q&A** – Generates high-quality questions & answers using Gemini AI  
✅ **Vector Search** – Uses FAISS for efficient retrieval  
✅ **User-Friendly UI** – Simple file upload and CSV export  
✅ **Lightweight & Fast** – Optimized for quick processing  

## 📌 Working  
1️⃣ Upload a PDF through the UI  
2️⃣ Analyze the document to generate questions & answers  
3️⃣ Download the CSV file with structured Q&A  

## 🛠️ Tech Stack  
**Flask** – Backend API  
**LangChain** – AI-powered question generation  
**FAISS** – Vector-based retrieval  
**Hugging Face** – Embeddings for document storage  
**HTML & CSS** – Simple frontend UI  
