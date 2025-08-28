# gemini-q-a-bot
A smart system that answers questions based on PDF content using Google Gemini 1.5 Flash and semantic search.

Features:
Upload any PDF
Chunk and embed content
Retrieve top-matching text
Ask questions in natural language
Get accurate, document-based answers

Tech Used:
PyMuPDF – Extract text from PDF
sentence-transformers – Embed chunks
scikit-learn – Cosine similarity
Google Generative AI – Gemini 1.5 Flash
NLTK – Sentence splitting

Setup:
pip install google-generativeai PyMuPDF sentence-transformers scikit-learn nltk
Set your API key:
os.environ["GOOGLE_API_KEY"] = "your-api-key"

Example Q&A:
Q: What are the in-patient hospitalization benefits?
A: Covers room rent, ICU, doctor fees, diagnostics, etc., based on policy conditions.
