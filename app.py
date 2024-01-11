import streamlit as st
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI 
from openai import OpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

st.title("Spotify Reviews QnA")

@st.cache_resource
def init_pinecone_index():
    """Initialize the Pinecone index.
    
    Returns:
        Pinecone: The Pinecone index.
    """
    pinecone.init(
        api_key=st.secrets["PINECONE_API_KEY"],
        environment=st.secrets["PINECONE_ENV"],
    )

    index_name = "spotify-reviews"
    embeddings = HuggingFaceEmbeddings()
    pinecone_index = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings
    )

    return pinecone_index

def _create_qa_chain(llm):
    """Creates a question answering chain with the given language model.
    
    Args:
        llm: The language model to use.
    
    Returns:
        BaseCombineDocumentChain: The question answering chain.
    """
    from langchain.chains.question_answering import load_qa_chain
    
    prompt_template = "Below are some reviews for our music streaming application called Spotify. Answer the question in the end based on the provided reviews. If none of the reviews are not relevant to the question, just say that you don't know, don't try to make up an answer. \n### \nReviews:  \n{context} \n### \nQuestion: \n{question}"
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return qa_chain

def _create_mq_retriever(llm, index, k=20):
    """Creates a multi-query retriever with the given language model and index.
    
    Args:
        llm: The language model to use.
        index: The index to use.
        k (int): The number of documents to retrieve.
    
    Returns:
        MultiQueryRetriever: The multi-query retriever.
    """
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=index.as_retriever(search_kwargs={"k": k}), llm=llm)
    
    mq_retriever.llm_chain.prompt.template = 'You are an AI language model assistant. Your task is to generate 3 different versions of queries from the user question to retrieve relevant reviews written for our music streaming application called Spotify. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search. Provide these alternative queries separated by newlines. \n### User question: {question}'
    
    return mq_retriever

def _flag_input(input):
    """Flag inappropriate input using OpenAI's Moderation API.

    Args:
        input (str): The input to flag.

    Returns:
        bool: True if the input is flagged, False otherwise.
    """
    client = OpenAI()
    response = client.moderations.create(input=input)
    return response.results[0].flagged

def _choose_top_reviews(docs, k):
    """Choose the top k reviews by likes.
    
    Args:
        docs (list): A list of documents.
        k (int): The number of documents to choose.
        
    Returns:
        list: A list of documents.
    """
    idx_likes = {idx: doc.metadata['review_likes'] for idx, doc in enumerate(docs)}
    
    sorted_idx_likes = {k: v for k, v in sorted(idx_likes.items(), key=lambda item: item[1], reverse=True)}
    top_k_idx = list(sorted_idx_likes.keys())[:k]
    
    return [docs[idx] for idx in top_k_idx]
    
def qa_pipeline(llm, index, question, k=40):
    """A question answering pipeline that takes a question and returns an answer synthesized from the relevant reviews retrieved from the index.

    Args:
        question (str): The question to answer.
        final_k (int): The number of reviews to retrieve from the index.

    Returns:
        _type_: _description_
    """
    qa_chain = _create_qa_chain(llm)
    retriever = _create_mq_retriever(llm, index, k=20)
    
    if _flag_input(question):
        return "Your question contains inappropriate content. Please try again."
    
    docs = retriever.get_relevant_documents(question)
    top_reviews = _choose_top_reviews(docs, k)
    
    result = qa_chain(
        {"input_documents": top_reviews, "question": question}, return_only_outputs=False
    )
        
    return result


st.session_state.index = init_pinecone_index()
st.session_state.llm = ChatOpenAI(model="gpt-4-1106-preview", openai_api_key=st.secrets['OPENAI_API_KEY'], temperature=0)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
if prompt := st.chat_input("Enter a question related to Spotify reviews!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    result = qa_pipeline(st.session_state.llm, st.session_state.index, prompt)
    response = result['output_text']
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})