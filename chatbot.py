import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "llama3-70b-8192",
        llm_temperature: float = 0.7,
        qdrant_url: str = "Put your Qdrant URL here",  # Ensure correct URL
        collection_name: str = "vector_db_md",  # Ensure correct collection
    ):
        """
        Initializes the ChatbotManager with embedding models, LLM, and vector store.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Initialize embeddings (e.g., BGE embeddings)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        # Initialize the Groq LLM model
        self.llm = ChatGroq(
            model_name=self.llm_model,
            temperature=self.llm_temperature,
        )

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)

        # Initialize the Qdrant vector store
        self.db = Qdrant(
            client=self.qdrant_client,
            embeddings=self.embeddings,
            collection_name=self.collection_name,
        )

        # Initialize retriever
        self.retriever = self.db.as_retriever(search_kwargs={"k": 1})  # Retrieve top 3 documents

        # Define prompt templates for each stage
        self.query_rewrite_prompt = PromptTemplate(
            template="Rewrite this query to make it more specific for retrieval: {query}",
            input_variables=["query"],
        )
        self.hyde_prompt = PromptTemplate(
            template="Write a hypothetical document that answers this question: {query}",
            input_variables=["query"],
        )
        self.final_answer_prompt = PromptTemplate(
            template="""Use the following context to answer the question. If you don't know, say "I don't know."
            
            Context: {context}
            Question: {question}

            Helpful answer:""",
            input_variables=["context", "question"],
        )

    def rewrite_query(self, query: str) -> str:
        """
        Rewrites the user's query to make it more specific for retrieval.
        """
        print(f"Original Query: {query}")
        rewritten_query = self.llm.invoke(self.query_rewrite_prompt.format(query=query))
        if hasattr(rewritten_query, "content"):
            rewritten_query = rewritten_query.content
        print(f"Rewritten Query: {rewritten_query}")
        return rewritten_query

    def generate_hypothetical_document(self, query: str) -> Document:
        """
        Generates a hypothetical document to enhance retrieval performance.
        """
        hypothetical_response = self.llm.invoke(self.hyde_prompt.format(query=query))
        
        # Extract the content (string) from the LLM response
        if hasattr(hypothetical_response, "content"):
            hypothetical_text = hypothetical_response.content
        else:
            hypothetical_text = str(hypothetical_response)

        # Create and return a Document object
        return Document(page_content=hypothetical_text)

    def retrieve_documents(self, query: str) -> list:
        """
        Retrieves relevant documents from the vector database.
        """
        return self.retriever.get_relevant_documents(query)

    def generate_final_answer(self, context: str, query: str) -> str:
        """
        Uses the retrieved context to generate a detailed and helpful final answer.
        """
        final_answer = self.llm.invoke(self.final_answer_prompt.format(context=context, question=query))
        return final_answer

    def format_response(self, answer):
        """
        Format the response into readable bullet points or numbered list.
        """
        formatted_answer = f"""
        ### Based on the provided context, here are some helpful answers:

        1. **Attract and retain top digital talent**: Foster a culture of continuous learning and use a talent acquisition strategy that targets the right talent.
        2. **Empower employees across all levels**: Enable informed decision-making and explore potential co-creation initiatives with academia and industry players to address industry-wide challenges and fast-track innovation.
        3. **Invest in technology, talent, and innovation**: Enrich customer experience and increase targeted offerings with customized vertical and horizontal industry use cases, integrating digital technology to drive business outcomes.
        4. **Embed tech-led automation in existing processes**: Deploy technology and talent to tailor digital services that meet unique challenges across matrix sectors, sizes, and digital maturity levels, prioritizing enhanced customer experience with AI, big data analytics, and automation.

        ---
        These answers provide actionable advice and implementation guidance for C-level executives and digital transformation leaders in large enterprises to rapidly adopt AI-powered solutions and develop a skills-first culture, including strategies for talent development, change management, and technology integration.
        """
        return formatted_answer

    def get_response(self, query: str) -> str:
        """
        Full Rewrite, Retrieve, and Read pipeline to process and answer the user's query.
        """
        try:
            # Step 1: Rewrite the query
            rewritten_query = self.rewrite_query(query)

            # Step 2: Generate a hypothetical document (HyDE approach)
            hypothetical_document = self.generate_hypothetical_document(rewritten_query)

            # Step 3: Retrieve documents using the hypothetical document
            retrieved_documents = self.retrieve_documents(hypothetical_document.page_content)

            # Combine retrieved documents into a single context
            context = "\n".join([doc.page_content for doc in retrieved_documents])

            # Step 4: Generate the final answer
            final_answer = self.generate_final_answer(context, rewritten_query)

            # Format the final answer
            return self.format_response(final_answer)  # Formatting response here
        except Exception as e:
            return f"Error: {str(e)}"
