import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.docx import partition_docx
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

# Load environment variables from .env
load_dotenv('.env')

class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        qdrant_url: str = "Put your Qdrant URL here",
        collection_name: str = "vector_db_md",
        use_two_step_split: bool = False,
        remove_stopwords: bool = True, 
        custom_stopwords: set = set()  
    ):
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.api_key = os.getenv('QDRANT_API_KEY')
        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.api_key)
        self._initialize_collection()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )
        self.use_two_step_split = use_two_step_split
        self.remove_stopwords = remove_stopwords
        self.custom_stopwords = custom_stopwords

    def _initialize_collection(self):
        try:
            existing_collections = self.qdrant_client.get_collections().collections
            if not any(col.name == self.collection_name for col in existing_collections):
                collection_config = models.VectorParams(size=384, distance=models.Distance.COSINE)
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=collection_config
                )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize collection in Qdrant: {e}")

    def _load_document(self, file_path: str):
        try:
            extension = os.path.splitext(file_path)[-1].lower()

            if extension == ".pdf":
                loader = UnstructuredPDFLoader(file_path)
                return loader.load()
            elif extension == ".pptx":
                return partition_pptx(file_path)
            elif extension == ".docx":
                return partition_docx(file_path)
            elif extension == ".txt":
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                return [{"text": text}]
            else:
                raise ValueError(f"Unsupported file type: {extension}")
        except Exception as e:
            raise ValueError(f"Error loading document {file_path}: {e}")

    def clean_text(self, text: str) -> str:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english')) if self.remove_stopwords else set()

        # Add custom stopwords
        stop_words.update(self.custom_stopwords)

        text = text.lower()
        text = re.sub(r'[^\w\s.,%$]', '', text)

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

        return ' '.join(words)

    def preprocess_documents(self, docs: list) -> list:
        for doc in docs:
            doc.page_content = self.clean_text(doc.page_content)
        return docs

    @staticmethod
    def extract_page_number(text: str) -> int:
        page_number_match = re.search(r'(?:page\s*)(\d+)', text, re.IGNORECASE)
        if page_number_match:
            return int(page_number_match.group(1))
        return None

    def create_embeddings(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        try:
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_path)[-1].lower()
            source_path = os.path.abspath(file_path)

            docs = self._load_document(file_path)
            if not docs:
                raise ValueError("No documents were loaded from the file.")

            docs = self.preprocess_documents(docs)

            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=1000, 
                chunk_overlap=250
            )
            splits = text_splitter.split_documents(docs)

            if self.use_two_step_split:
                secondary_splitter = SentenceTransformersTokenTextSplitter(
                    model_name="sentence-transformers/all-MiniLM-L6-v2", 
                    chunk_size=512
                )
                splits = secondary_splitter.split_documents(splits)

            if not splits:
                raise ValueError("No text chunks were created from the documents.")

            smz_docs = ""
            for i, split in enumerate(splits):
                extracted_page = self.extract_page_number(split.page_content) or (i + 1)

                split.metadata = {
                    "file_name": file_name,
                    "file_type": file_extension,
                    "source": source_path,
                    "page": extracted_page,
                }

                smz_docs += f"\n\nDocument {i+1}:"
                smz_docs += f"\nPage: {split.metadata.get('page', 'N/A')} and Source: {split.metadata.get('source', 'Unknown')}"
                smz_docs += f"\nContent: {split.page_content[:100]}..."

            try:
                Qdrant.from_documents(
                    splits,
                    self.embeddings,
                    url=self.qdrant_url,
                    api_key=self.api_key,
                    prefer_grpc=False,
                    collection_name=self.collection_name
                )
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Qdrant: {e}")

            return f"Vector DB successfully created and stored in Qdrant for {file_name}!"
        except Exception as e:
            raise Exception(f"Error during embedding creation for {file_path}: {e}")
