import os
import json
import datetime
import subprocess
import assemblyai as aai
from pytubefix import YouTube
from fpdf import FPDF
import os
from dotenv import load_dotenv
import boto3
from langchain_community.vectorstores import FAISS
from FlagEmbedding import BGEM3FlagModel
from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
from langchain_aws.chat_models import ChatBedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# === SETUP ===
aai.settings.api_key = "8a010ac1fdb74105bca2aa651b32a3e2"  # Replace for production

load_dotenv()

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
)

class BGE_M3_Embedder:
    def __init__(self):
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def embed(self, texts: list[str]) -> list[list[float]]:
        print("🔍 Embedding with BGE-M3...")
        return self.model.encode(texts, max_length=8192)["dense_vecs"]

bge_m3_embedder = BGE_M3_Embedder()

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return True
    except FileNotFoundError:
        print("ffmpeg not found.")
        return False

def download_audio(url):
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
    filename = f"temp_audio_{yt.video_id}.mp3"
    file = stream.download(filename=filename)
    return file

def convert_to_wav(input_file):
    output_file = input_file.replace('.mp3', '.wav')
    cmd = ['ffmpeg', '-y', '-i', input_file, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', output_file]
    subprocess.run(cmd, check=True)
    return output_file

def transcribe_with_assemblyai(file_path):
    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(file_path)
    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")
    return transcript.text

def process_with_claude(transcript):
    system_prompt = """
You are an AI assistant helping create structured training content from expert (SME) video transcripts.

Your goal is to convert the following transcript into a clean, structured learning document that can be used to train employees. 
Organize the output with the following fields:
 
{
  "Title": "",
  "Objective": "",
  "TargetAudience": "",
  "KeyConcepts": ["", "", ...],
  "ToolsMentioned": ["", "", ...],
  "StepByStepInstructions": [
    {"StepNumber": 1, "Instruction": "", "Purpose": "", "Example": ""}
  ],
  "ImportantNotes": ""
}

Be precise, instructional, and clear. Use professional tone suitable for internal corporate learning platforms.

Transcript:
"""
    input_data = {
        "messages": [
            {"role": "user", "content": system_prompt + transcript}
        ],
        "max_tokens": 20000,
        "temperature": 0,
        "anthropic_version": "bedrock-2023-05-31"
    }

    response = bedrock_client.invoke_model(
        modelId="arn:aws:bedrock:us-east-1:225989333617:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=json.dumps(input_data),
        contentType="application/json"
    )
    response_body = json.loads(response['body'].read().decode())
    content = response_body.get("content", [])
    return content[0]["text"] if content and "text" in content[0] else None

class BGEWrapper(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts)['dense_vecs']

    def embed_query(self, text):
        return self.model.encode([text])['dense_vecs'][0]

def embed_document(content):
    print("🔍 Splitting and embedding document...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents([Document(page_content=content)])
    texts = [chunk.page_content for chunk in chunks]

    bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    embedding = BGEWrapper(bge_model)

    db = FAISS.from_texts(texts=texts, embedding=embedding)
    return db



def run_chat(db):
    retriever = db.as_retriever()
    qa_model = ChatBedrock(
        client=bedrock_client,
        model_id="arn:aws:bedrock:us-east-1:225989333617:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        provider="anthropic"
    )
    qa_chain = RetrievalQA.from_chain_type(llm=qa_model, retriever=retriever)
    print("\n🧠 Chat ready. Type your questions (or type 'exit' to stop):")
    while True:
        q = input("Q: ")
        if q.lower() in ['exit', 'quit']:
            break
        try:
            a = qa_chain.invoke(q)
            print(f"A: {a}\n")
        except Exception as e:
            print(f"Error during retrieval: {e}")

def cleanup(files):
    for f in files:
        if os.path.exists(f):
            os.remove(f)

def main():
    if not check_ffmpeg():
        return

    print("🎥 Choose input method:")
    print("1. YouTube URL")
    print("2. Local audio file from 'inputs' folder")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        url = input("Enter YouTube URL: ")
        audio_file = download_audio(url)
        wav_file = convert_to_wav(audio_file)

    elif choice == "2":
        print("\nAvailable files in 'inputs/' folder:")
        files = [f for f in os.listdir("inputs") if f.lower().endswith((".mp3", ".wav", ".mp4", ".mkv", ".mov"))]
        if not files:
            print("No audio files found in 'inputs' folder.")
            return

        for idx, file in enumerate(files, 1):
            print(f"{idx}. {file}")
        file_choice = input("Select a file number: ").strip()

        try:
            selected = files[int(file_choice) - 1]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return

        input_path = os.path.join("inputs", selected)
        audio_file = None

        if selected.lower().endswith(".mp3"):
            wav_file = convert_to_wav(input_path)
            audio_file = input_path

        elif selected.lower().endswith(".wav"):
            wav_file = input_path  # Already WAV, use directly

        else:
            # Video file – extract audio using ffmpeg
            wav_file = os.path.splitext(input_path)[0] + ".wav"
            cmd = ['ffmpeg', '-y', '-i', input_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', wav_file]
            subprocess.run(cmd, check=True)


    else:
        print("Invalid choice.")
        return

    print("\n🎙️ Transcribing...")
    transcript = transcribe_with_assemblyai(wav_file)

    print("\n📄 Generating structured process doc...")
    structured_doc = process_with_claude(transcript)
    print("\n--- Claude Output ---\n", structured_doc[:800], "\n...\n")

    # Save to docs/ folder
    os.makedirs("docs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("docs", f"structured_doc_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(structured_doc)

    print(f"✅ Structured document saved to: {output_path}")

    print("🔐 Creating embeddings and vector DB...")
    db = embed_document(structured_doc)

    run_chat(db)

    if audio_file:
        cleanup([audio_file, wav_file])
    elif wav_file.endswith(".wav"):
        cleanup([wav_file])

if __name__ == "__main__":
    main()
