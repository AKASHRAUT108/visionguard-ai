import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class RAGEngine:
    def __init__(self):
        # Load and split documents
        loader = TextLoader("data/knowledge_base.txt", encoding='utf-8')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = FAISS.from_documents(texts, embeddings)

        # Set up the LLM (using HuggingFace Hub – requires token)
        self.llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
            model_kwargs={"temperature": 0.1, "max_length": 500}
        )

        # Create a prompt template
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Use the following context to answer the question. If you don't know, say you don't know.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def ask(self, question: str):
        if not question:
            return "No question provided."

        # Retrieve relevant documents
        docs = self.db.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate answer
        answer = self.chain.run(context=context, question=question)
        return answer