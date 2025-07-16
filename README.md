🎬 Enhanced Video QA with Reranker & Clip Extraction
This repo allows you to run a question-answering (QA) pipeline on timestamped transcripts of videos, generate high-quality answers using Claude via AWS Bedrock, and extract supporting video clips from the relevant timestamp ranges.

🚀 How It Works
🎥 Transcribe a video file into timestamped sentence-level chunks.

🤖 Run a QA query against the transcript using:

Semantic embedding search

Reranker model

Claude LLM for answer synthesis

🎬 Extract video clips that match the timestamps of top-ranked context.

📂 Folder Structure
bash
Copy
Edit
.
├── timestampsentence.py            # Step 1: Transcript generator
├── enhanced_qa_with_reranker.py   # Step 2: QA engine with Claude & reranker
├── clip_manager.py                # Video clip extraction logic
├── inputs/                        # Raw video files (optional, default path)
├── docs/                          # QA logs and debug metadata
├── generated_clips/              # Output folder for final mp4 clips
├── requirements.txt              # Python dependencies
🧪 Quickstart
1️⃣ Step 1: Transcribe the video
Make sure your video file is in the root folder (or specify path inside the script).

bash
Copy
Edit
python timestampsentence.py
➡️ This generates a JSON file named:

php-template
Copy
Edit
<your_video>_sentences.json
2️⃣ Step 2: Run QA with reranker + Claude
Ensure both the .mp4 and generated .json are in the root folder.

bash
Copy
Edit
python enhanced_qa_with_reranker.py
📦 Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
🔐 Ensure your .env file includes valid AWS credentials for Bedrock (Claude).

💡 Tips
🧠 Your transcript and video must share the same base filename.

📁 All files must be accessible in the working directory (or edit paths accordingly).

🛠️ The system uses:

BGE-M3 for embedding

BGE reranker for better chunk selection

Claude Sonnet for answer generation

FFmpeg for clip slicing
