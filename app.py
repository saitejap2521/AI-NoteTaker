import os
import torch
import gradio as gr
from pathlib import Path
from openai import OpenAI
from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# Environment variables
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACE_API_KEY")
login(hf_token, add_to_git_credential=True)


openai_client = OpenAI(api_key=OPENAI_KEY)


chatbot = pipeline(
    model="meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=128001
    )


def chat(messages):
  ai_response = chatbot(messages, max_new_tokens=50)[0]['generated_text'][-1]['content']
  return ai_response



def transcribe_audio(audio):
  audio_file = Path(audio)
  transcriber = openai_client.audio.transcriptions.create(
      model="whisper-1",
      file=audio_file
  )

  transcription = transcriber.text
  return transcription


def notes_from_audio(audio=None):
  transcription = transcribe_audio(audio)

  SYSTEM_PROMPT_MESSAGE = "You are an expert notetaker, use the following transcription from my audio to generate"
  SYSTEM_PROMPT_MESSAGE += " notes based on it. Make sure to keep these brief using bullet points."
  prompt = SYSTEM_PROMPT_MESSAGE + "/nTranscription/n" + transcription
  messages = [{'role':'user','content':prompt}]
  notes = chat(messages)
  return notes, transcription


with gr.Blocks() as demo:
  with gr.Row():
    audio_input = gr.Audio(sources=["upload","microphone"], type="filepath", label="Audio Transcriber")

  with gr.Row():
    transcription = gr.Textbox(label="Transcription")
    notes = gr.Textbox(label="Notes")

  with gr.Row():
    submit_button = gr.Button("Start")
    submit_button.click(fn = notes_from_audio, inputs=[audio_input], outputs=[notes, transcription])

demo.launch()