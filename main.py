from fastapi import FastAPI, File, UploadFile
import uvicorn
from fastapi.responses import JSONResponse
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from io import BytesIO
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and processor
pretrain_model_location = r'C:\Users\Anning\Music\model\New folder\New folder\whisper-small_Akan'
processor = WhisperProcessor.from_pretrained('openai/whisper-small', language='yo', task='transcribe')
model = WhisperForConditionalGeneration.from_pretrained(pretrain_model_location)

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Whisper Transcription API!"}

@app.get("/ping")
async def ping():
    return "Hello server is working"

@app.post("/transcribe")
async def transcribe(files: UploadFile = File(...)):
    if not files.content_type.startswith("audio/"):
        return JSONResponse(content={"error": "Invalid file type. Please upload an audio file."}, status_code=400)

    try:
        audio_bytes = await files.read()

        audio = AudioSegment.from_file(BytesIO(audio_bytes))
        audio = audio.set_channels(1).set_frame_rate(16000)

        raw_data = audio.raw_data
        audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        audio_np /= np.iinfo(np.int16).max

        input_features = processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return JSONResponse(content={"transcription": transcription[0]})
    except Exception as e:
        return JSONResponse(content={"error": f"Error during transcription: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
