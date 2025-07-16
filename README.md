🎬 Enhanced Video QA with Reranker & Clip Extraction
This repo allows you to run a question-answering (QA) pipeline on timestamped transcripts of videos, generate answers using Claude (via AWS Bedrock), and extract video clips that support the answers.

🚀 How It Works
Transcribe a video file into timestamped sentence-level chunks.

Run a QA query against the transcript using semantic search + reranker + Claude LLM.

Extract supporting video clips from the timestamps of the best-matching chunks.

📂 Folder Structure

.
├── timestampsentence.py            # Step 1: Transcript generator
├── enhanced_qa_with_reranker.py   # Step 2: Main QA engine with reranker + Claude
├── clip_manager.py                # Video clip extraction
├── inputs/                        # Folder for raw video files
├── docs/                          # Output metadata and optional QA logs
├── generated_clips/              # Output folder for final mp4 evidence clips
├── requirements.txt              # Python dependencies

🧪 Quickstart

1️⃣ Step 1: Transcribe the video

Make sure video file is in root folder.

python timestampsentence.py 

This will generate a JSON file.

2️⃣ Step 2: Run QA with reranker + Claude

python enhanced_qa_with_reranker.py 

With both video and JSON file in root folder.

📦 Install Dependencies

pip install -r requirements.txt
Make sure your .env has valid AWS credentials for Bedrock + Claude access.

💡 Tips
Make sure video and transcript are named consistently and in the same folder.
