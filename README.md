# TEXT-SUMMARIZATION

![Python](https://img.shields.io/badge/Python-3.7.6-blueviolet)
![Frontend](https://img.shields.io/badge/Frontend-Streamlit-green)

Generate summaries from texts using Streamlit & HuggingFace Pipeline

Extract text from url and divide text into chunks if length of text is more than 500 words

For summarization, only the following formats are accepted:
- Raw text entered in text box 
- URL of an article to be summarized 
- Documents with .txt, .pdf or .docx file formats


Using extractive summarization, the app identifies salient information and groups it together into a concise summary. 
Using the app, longer documents or texts will be divided into chunks and summarized.
You can choose between two models:
- Facebook-Bart, trained on large CNN Daily Mail articles
- Sshleifer-Distilbart, which is a distilled version of the large Bart model

****It is important to note that long documents will take longer to generate summaries.****

