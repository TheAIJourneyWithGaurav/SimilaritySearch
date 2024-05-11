#loader---> CSV loader
from langchain_community.document_loaders.csv_loader import CSVLoader
loader=CSVLoader("a.csv")
data=loader.load()
#embeddings---> Huggingface embeddings(sentence transformer)
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",model_kwargs={'device':'cpu'})
#vector db ---> FAISS 
from langchain_community.vectorstores import FAISS
db=FAISS.from_documents(data,embeddings)
#semilarity search(QUERY)
query=input("enter query: ")

retreival=db.similarity_search(query)
print(retreival[0].page_content.strip())
