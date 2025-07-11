import os
import assemblyai as aai

# Set your AssemblyAI API key
aai.settings.api_key = "8a010ac1fdb74105bca2aa651b32a3e2"  # ← Replace with your actual key

# Supported formats
SUPPORTED_EXTS = (".mp3", ".wav", ".mp4", ".m4a", ".mov", ".aac")

# List files in root
files = [f for f in os.listdir('.') if f.lower().endswith(SUPPORTED_EXTS)]

if not files:
    print("❌ No supported audio/video files found in root directory.")
    exit(1)

print("🎵 Available files:")
for i, f in enumerate(files, 1):
    print(f"{i}. {f}")

try:
    selection = int(input("Select a file number: ").strip())
    filename = files[selection - 1]
except (ValueError, IndexError):
    print("❌ Invalid selection.")
    exit(1)

# Transcribe with AssemblyAI
print(f"🎙️ Transcribing '{filename}' ...")
transcriber = aai.Transcriber()
config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)
transcript = transcriber.transcribe(filename, config)

if transcript.status == aai.TranscriptStatus.error:
    print(f"❌ Transcription failed: {transcript.error}")
    exit(1)

# Save as .txt
output_file = os.path.splitext(filename)[0] + "_transcript.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(transcript.text)

print(f"✅ Transcript saved to: {output_file}")

