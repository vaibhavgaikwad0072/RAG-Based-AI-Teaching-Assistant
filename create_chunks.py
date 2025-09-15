import whisper
import json
import os

# use a smaller model for CPU
model = whisper.load_model("small")

os.makedirs("jsons", exist_ok=True)
audios = os.listdir("audios")

for audio in audios: 
    if "_" in audio:
        number = audio.split("_")[0]
        title = audio.split("_")[1].replace(".mp3", "").replace(".wav", "").replace(".m4a", "").replace(".mp4", "")

        print(f"Processing: {number} {title}")

        result = model.transcribe(
            f"audios/{audio}",
            language="hi",
            task="translate",
            fp16=False  
        )
        
        chunks = [
            {
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            for segment in result["segments"]
        ]

        chunks_with_metadata = {
            "chunks": chunks,
            "text": result["text"]
        }

        output_path = f"jsons/{audio}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved → {output_path}")
