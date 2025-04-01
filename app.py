import re
import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter



## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')
if "GROQ_API_KEY" not in st.secrets:
    st.error("Groq API Key is missing. Please add it to Streamlit Cloud Secrets.")
else:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

generic_url = st.text_input("URL", label_visibility="collapsed")
def convert_youtube_url(url):
    """Converts shortened YouTube URLs (youtu.be) to standard format and extracts video ID."""
    if not url:
        return None  # Return None if the URL is empty or None

    # Match short URL format (youtu.be)
    match = re.match(r"https?://youtu\.be/([a-zA-Z0-9_-]+)", url)
    if match:
        video_id = match.group(1)
        return video_id  # Return only the video ID
    
    # Match full YouTube URL format (youtube.com/watch?v=...)
    match = re.match(r"https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)  # Return only the video ID

    return None  # If not a valid YouTube URL, return None
video_id = convert_youtube_url(generic_url)

## Gemma Model USsing Groq API
llm =ChatGroq(model="llama3-8b-8192", groq_api_key=GROQ_API_KEY)

#prompt_template="""
#Provide a summary of the following content in 300 words:
#Content:{text}

#"""
#prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

map_prompt = PromptTemplate(
    template="""
    Summarize the following chunk of text in 100 words:
    {text}
    """,
    input_variables=["text"],
)

combine_prompt = PromptTemplate(
    template="""
    Combine the following summaries into a final summary of 500 words:
    {text}
    """,
    input_variables=["text"],
)

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not GROQ_API_KEY.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                if video_id:
                    try:
                        # Fetch transcript for the video ID
                        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                        transcript_text = " ".join([entry['text'] for entry in transcript])
                        
                        # Split the transcript text into smaller chunks
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        chunks = text_splitter.split_text(transcript_text)
                        
                        docs = [Document(page_content=chunk, metadata={"source": generic_url}) for chunk in chunks]
                        st.write("✅ Transcript fetched and split successfully!")
                        
                    except Exception as e:
                        st.error(f"Error fetching transcript: {e}")
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs=loader.load()

                ## Chain For Summarization
                if docs:
                    chain=load_summarize_chain(llm=llm,chain_type="map_reduce",map_prompt=map_prompt,combine_prompt=combine_prompt)
                    output_summary = chain.run(docs)
                    st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
                    
