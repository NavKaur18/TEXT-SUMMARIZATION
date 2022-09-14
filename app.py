import validators, re
from fake_useragent import UserAgent
from bs4 import BeautifulSoup   
import streamlit as st
from transformers import pipeline
import time
import base64 
import requests
import docx2txt
from io import StringIO
from PyPDF2 import PdfFileReader
import warnings
warnings.filterwarnings("ignore")


time_str = time.strftime("%d%m%Y-%H%M%S")
#Functions

def article_text_extractor(url: str):
    
    '''Extract text from url and divide text into chunks if length of text is more than 500 words'''
    
    ua = UserAgent()

    headers = {'User-Agent':str(ua.chrome)}

    r = requests.get(url,headers=headers)
    
    soup = BeautifulSoup(r.text, "html.parser")
    title_text = soup.find_all(["h1"])
    para_text = soup.find_all(["p"])
    article_text = [result.text for result in para_text]
    article_header = [result.text for result in title_text][0]
    article = " ".join(article_text)
    article = article.replace(".", ".<eos>")
    article = article.replace("!", "!<eos>")
    article = article.replace("?", "?<eos>")
    sentences = article.split("<eos>")
    
    current_chunk = 0
    chunks = []
    
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(" ")) <= 600:
                chunks[current_chunk].extend(sentence.split(" "))
            else:
                current_chunk += 1
                chunks.append(sentence.split(" "))
        else:
            print(current_chunk)
            chunks.append(sentence.split(" "))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = " ".join(chunks[chunk_id])

    return article_header, chunks
    
def preprocess_plain_text(x):

    x = x.encode("ascii", "ignore").decode()  # unicode
    x = re.sub(r"https*\S+", " ", x)  # url
    x = re.sub(r"@\S+", " ", x)  # mentions
    x = re.sub(r"#\S+", " ", x)  # hastags
    x = re.sub(r"\s{2,}", " ", x)  # over spaces
    x = re.sub("[^.,!?A-Za-z0-9]+", " ", x)  # special charachters except .,!?

    return x

def extract_pdf(file):
    
    '''Extract text from PDF file'''
    
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_text += page.extractText()

    return all_text


def extract_text_from_file(file):
    
    '''Extract text from uploaded file'''

    # read text file
    if file.type == "text/plain":
        # To convert to a string based IO:
        stringio = StringIO(file.getvalue().decode("utf-8"))

        # To read file as string:
        file_text = stringio.read()

    # read pdf file
    elif file.type == "application/pdf":
        file_text = extract_pdf(file)

    # read docx file
    elif (
        file.type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        file_text = docx2txt.process(file)

    return file_text

def summary_downloader(raw_text):
    
	b64 = base64.b64encode(raw_text.encode()).decode()
	new_filename = "new_text_file_{}_.txt".format(time_str)
	st.markdown("#### Download Summary as a File ###")
	href = f'<a href="data:file/txt;base64,{b64}" download="{new_filename}">Click to Download!!</a>'
	st.markdown(href,unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def facebook_model():
    
    summarizer = pipeline('summarization',model='facebook/bart-large-cnn')
    return summarizer
    
@st.cache(allow_output_mutation=True)
def schleifer_model():
    
    summarizer = pipeline('summarization',model='sshleifer/distilbart-cnn-12-6')
    return summarizer
    
#Streamlit App  
st.markdown(
    "<h1 style='text-align: center; color: blue;'>Text Summarizer</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align: center; color: black;'>from URL,Article and by Uploading Files 📝</h4>",
    unsafe_allow_html=True,
)

model_type = st.selectbox(
    "Select Model", options=["Facebook-Bart", "Sshleifer-DistilBart"]
)
# model_type = st.selectbox('Select the model', ('BART', 'T5'))


st.markdown(
    "For summarization, only the following formats are accepted: "
)
st.markdown(
    """- Raw text entered in text box 

- URL of an article to be summarized 

- Documents with .txt, .pdf or .docx file formats"""
)
with st.expander("See explanation"):
     st.write( """The app supports extractive summarization which aims to identify the salient information that is then extracted and grouped together to form a concise summary. 

    For documents or text that is more than 500 words long, the app will divide the text into chunks and summarize each chunk.

    There are two models available to choose from:

    

    - Facebook-Bart, trained on large CNN Daily Mail articles

    - Sshleifer-Distilbart, which is a distilled version of the large Bart model

     

    Please do note that the model will take longer to generate summaries for documents that are too long""")

st.markdown("---")

url_text = st.text_input("Please Enter a url here")


st.markdown(
    "<h3 style='text-align: center; color: blue;'>OR</h3>",
    unsafe_allow_html=True,
)

plain_text = st.text_input("Please Paste/Enter plain text here")

st.markdown(
    "<h3 style='text-align: center; color: blue;'>OR</h3>",
    unsafe_allow_html=True,
)

upload_doc = st.file_uploader(
    "Upload a .txt, .pdf, .docx file for summarization"
)

is_url = validators.url(url_text)

if is_url:
    # complete text, chunks to summarize (list of sentences for long docs)
    article_title,chunks = article_text_extractor(url=url_text)
    
elif upload_doc:
    
    clean_text = preprocess_plain_text(extract_text_from_file(upload_doc))

else:
    
    clean_text = preprocess_plain_text(plain_text)

# summarize = st.button("Summarize")
col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    summarize = st.button("Summarize")

if summarize:
    if model_type == "Facebook-Bart":
        if is_url:
            text_to_summarize = chunks
        else:
            text_to_summarize = clean_text
    # extractive summarizer

        with st.spinner(
            text="Loading Facebook-Bart Model and Extracting summary. This might take a few seconds depending on the length of your text..."
        ):
            summarizer_model = facebook_model()
            summarized_text = summarizer_model(text_to_summarize, max_length=100000, min_length=30)
            summarized_text = ' '.join([summ['summary_text'] for summ in summarized_text])


            # final summarized output
            st.subheader("Summarized text")
            

            if is_url:
            
                # view summarized text (expander)
                st.markdown(f"Article title: {article_title}")
                
            st.write(summarized_text)
            
            summary_downloader(summarized_text)

    elif model_type == "Sshleifer-DistilBart":
            if is_url:
                text_to_summarize = chunks
            else:
                text_to_summarize = clean_text
        # extractive summarizer

            with st.spinner(
                text="Loading Sshleifer-DistilBart Model and Extracting summary. This might take a few seconds depending on the length of your text..."
            ):
                summarizer_model = schleifer_model()
                summarized_text = summarizer_model(text_to_summarize, max_length=100, min_length=30)
                summarized_text = ' '.join([summ['summary_text'] for summ in summarized_text])       

            # final summarized output
            st.subheader("Summarized text")
            
            if is_url:
            
                # view summarized text (expander)
                st.markdown(f"Article title: {article_title}")
                
            st.write(summarized_text)
            
            summary_downloader(summarized_text)

