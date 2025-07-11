import os
import json
import assemblyai as aai
import datetime

# === Set API Key ===
aai.settings.api_key = "8a010ac1fdb74105bca2aa651b32a3e2"  # Replace this

# === Select File from Directory ===
SUPPORTED_EXTS = (".mp3", ".wav", ".mp4", ".m4a", ".mov", ".aac")
files = [f for f in os.listdir('.') if f.lower().endswith(SUPPORTED_EXTS)]

if not files:
    print("❌ No supported audio/video files found.")
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

# === Transcription Config ===
config = aai.TranscriptionConfig(
    speaker_labels=False,
    auto_chapters=False,
    auto_highlights=False
)

# === Transcribe ===
print(f"🎙️ Transcribing '{filename}'...")
transcriber = aai.Transcriber(config=config)
transcript = transcriber.transcribe(filename)

if transcript.status == "error":
    raise RuntimeError(f"Transcription failed: {transcript.error}")

# === Extract Sentences with Timestamps ===
sentences = transcript.get_sentences()
output_sentences = []

for sentence in sentences:
    output_sentences.append({
        "text": sentence.text,
        "start": str(datetime.timedelta(milliseconds=sentence.start)),
        "end": str(datetime.timedelta(milliseconds=sentence.end)),
        "confidence": sentence.confidence
    })

# === Save to JSON ===
output_file = os.path.splitext(filename)[0] + "_sentences.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_sentences, f, indent=2)

print(f"✅ Sentence transcript saved to: {output_file}")
