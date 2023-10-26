import openai
# Set your OpenAI API key
with open('api_key.txt', 'r') as file:
    openai.api_key = file.read().strip()
    
#Part1:Data loading
# Step1: PDF
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("2023Catalog.pdf")
pages = loader.load()
len(pages)
page = pages[0]
print(page.page_content[0:500])
page.metadata

# Step2: Youtube
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
url="https://www.youtube.com/watch?v=kuZNIvdwnMc"
save_dir="./docs/youtube/"

loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    OpenAIWhisperParser()
)

docs = loader.load()
print(docs[0].page_content[0:500])

# Step3: URLs
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.sfbu.edu/admissions/student-health-insurance")
docs = loader.load()
print(docs[0].page_content[:500])

# Part2: Embeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
     chunk_size = 1500,
     chunk_overlap = 150
 )
splits = text_splitter.split_documents(docs)

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

from langchain.vectorstores import Chroma
persist_directory = './docs/chroma/'

# remove old database files if any
# get_ipython().system('rm -rf ./docs/chroma')  

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
print(vectordb._collection.count())

question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question,k=3)
print("The original answer: ")
print("===================>")
print(docs[0].page_content)
print("===================>")
vectordb.persist()

docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)
print("The diverse answer1: ")
print("===================>")
print(docs_mmr[0].page_content)
print("===================>")
print("The diverse answer2: ")
print("===================>")
print(docs_mmr[1].page_content)
print("===================>")

new_question = "What are the application requirements for the MSCS program at SFBU?"
# docs = vectordb.similarity_search(
#     question,
#     k=3,
#     filter={"source":
#      "2023Catalog.pdf"}
# )
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [

 AttributeInfo(
   name="source",
   description="The catalog the chunk is from, should \
      be `2023Catalog.pdf`",
   type="string",
   ),

 AttributeInfo(
   name="page",
   description="The page from the catalog",
   type="integer",
 ),

]

document_content_description = "Lecture notes"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)

docs = retriever.get_relevant_documents(new_question)
for d in docs:
    print(d.metadata)
