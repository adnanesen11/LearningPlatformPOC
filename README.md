🎬 Enhanced Video QA with Reranker & Clip Extraction
This repo allows you to run a question-answering (QA) pipeline on timestamped transcripts of videos, generate high-quality answers using Claude via AWS Bedrock, and extract supporting video clips from the relevant timestamp ranges.

📂 Folder Structure

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

python timestampsentence.py
➡️ This generates a JSON file.

2️⃣ Step 2: Run QA with reranker + Claude
Ensure both the .mp4 and generated .json are in the root folder.

python enhanced_qa_with_reranker.py

📦 Install Dependencies

pip install -r requirements.txt

🔐 Ensure your .env file includes valid AWS credentials for Bedrock (Claude).

💡 Tips
🧠 Your transcript and video must share the same base filename.

📁 All files must be accessible in the working directory (or edit paths accordingly).

