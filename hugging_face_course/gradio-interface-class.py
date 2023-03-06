import gradio as gr
import numpy as np
from transformers import pipeline

# Simple Gradio example with audio


# An audio-to-audio function that takes an audio file and reverses it
def reverse_audio(audio):
    # We receive the input as a numpy array which is returned here as a
    # tuple of `(sample_rate, data)`
    sr, data = audio
    reversed_audio = (sr, np.flipud(data))
    return reversed_audio


# Create the audio component where the source is the mic
mic = gr.Audio(source="microphone", type="numpy", label="Speak here...")
# gr.Interface(reverse_audio, mic, "audio").launch()


# In the following example we have a function that takes a dropdown index, a slider value,
# a number, and returns the audio sample of a musical tone

# We pass a list of components which correspond to a parameter, and a list of output
# components which correspond to a returned value
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def generate_tone(note, octave, duration):
    sr = 48_000
    a4_freq, tones_from_a4 = 440, 12 * (octave - 4) + (note - 9)
    frequency = a4_freq * (2 ** (tones_from_a4 / 12))
    duration = int(duration)
    audio = np.linspace(0, duration, duration * sr)
    audio = (20_000 * np.sin(audio * (2 * np.pi * frequency))).astype(np.int16)
    return (sr, audio)


"""
gr.Interface(
    generate_tone,
    [
        gr.Dropdown(notes, type="index"),
        gr.Slider(minimum=4, maximum=6, step=1),
        gr.Textbox(type="text", value=1, label="Duration in seconds"),
    ],
    "audio",
).launch()
"""


# Here is an interface that demos a speech-recognition model. It accepts either a mic
# input or an uploaded file.
# The `transcribe_audio` function processes the audio and returns the transcription
model = pipeline("automatic-speech-recognition")


def transcribe_audio(mic=None, file=None):
    if mic is not None:
        audio = mic
    elif file is not None:
        audio = file
    else:
        return "You must either provide a mic recording or a file"
    transcription = model(audio)["text"]
    return transcription


gr.Interface(
    fn=transcribe_audio,
    inputs=[
        gr.Audio(source="microphone", type="filepath", optional=True),
        gr.Audio(source="upload", type="filepath", optional=True),
    ],
    outputs="text",
).launch()
