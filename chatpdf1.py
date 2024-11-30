import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def obter_texto_pdf(arquivos_pdf):
    texto = ""
    for pdf in arquivos_pdf:
        leitor_pdf = PdfReader(pdf)
        for pagina in leitor_pdf.pages:
            texto += pagina.extract_text()
    return texto


def obter_pedacos_texto(texto):
    divisor_texto = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    pedacos = divisor_texto.split_text(texto)
    return pedacos


def obter_armazenamento_vetorial(pedacos_texto):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    armazenamento_vetorial = FAISS.from_texts(pedacos_texto, embedding=embeddings)
    armazenamento_vetorial.save_local("faiss_index")


def obter_cadeia_conversacional():
    modelo_prompt = """
    Responda à pergunta o mais detalhadamente possível com base no contexto fornecido. Certifique-se de fornecer todos os detalhes. Se a resposta não estiver no contexto fornecido, apenas diga "a resposta não está disponível no contexto", não forneça uma resposta errada.\n\n
    Contexto:\n {context}?\n
    Pergunta: \n{question}\n

    Resposta:
    """

    modelo = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=modelo_prompt, input_variables=["context", "question"])
    cadeia = load_qa_chain(modelo, chain_type="stuff", prompt=prompt)

    return cadeia


def entrada_usuario(pergunta_usuario):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    novo_db = FAISS.load_local("faiss_index", embeddings)
    documentos = novo_db.similarity_search(pergunta_usuario)

    cadeia = obter_cadeia_conversacional()

    resposta = cadeia(
        {"input_documents": documentos, "question": pergunta_usuario},
        return_only_outputs=True)

    print(resposta)
    st.write("Resposta: ", resposta["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Converse com PDF usando Gemini💁")

    pergunta_usuario = st.text_input("Faça uma Pergunta sobre os Arquivos PDF")

    if pergunta_usuario:
        entrada_usuario(pergunta_usuario)

    with st.sidebar:
        st.title("Menu:")
        arquivos_pdf = st.file_uploader("Carregue seus Arquivos PDF e Clique no Botão Enviar e Processar",
                                        accept_multiple_files=True)
        if st.button("Enviar e Processar"):
            with st.spinner("Processando..."):
                texto_bruto = obter_texto_pdf(arquivos_pdf)
                pedacos_texto = obter_pedacos_texto(texto_bruto)
                obter_armazenamento_vetorial(pedacos_texto)
                st.success("Concluído")


if __name__ == "__main__":
    main()