import os
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pynput import keyboard
from huggingsound import SpeechRecognitionModel

# Choose the sampling frequency
fs = 44100

# Initialize global variables
recording = []
is_recording = False
stream = None
speech_recognition_model = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

# Initialize the service
def init():
    # Load the model
    global model
    model = load_model(speech_recognition_model)

    # Start a keyboard listener that runs in a separate thread
    global listener
    with keyboard.Listener(on_press=on_press) as listener: # type: ignore
        listener.join()  # Wait for the listener to finish (i.e. for the spacebar key to be pressed)


# Load the model
def load_model(model_name):
    model = SpeechRecognitionModel(model_name)
    return model

# Transcribe the audio
def transcribe(model, audio_file):
    transcriptions = model.transcribe([audio_file])
    return transcriptions["transcription"]

# Define a callback function to record audio
def audio_callback(indata, frames, time, status):
    if is_recording:
        recording.append(indata.copy())

# Define a function that starts or stops recording when the spacebar key is pressed
def on_press(key):
    global is_recording
    global stream
    global model
    global listener

    stream = sd.InputStream(samplerate=fs, channels=1, callback=audio_callback)
    if key == keyboard.Key.space:
        if not is_recording:
            # Print message that recording has started
            print("Recording started...")

            # Start recording audio from the default microphone
            is_recording = True
            stream.start()
        else:
            # Stop recording
            is_recording = False
            stream.stop()
            stream.close()

            # Save the recording to a wav file
            audio_data = np.concatenate(recording, axis=0)
            output_file = 'output.wav'
            write(output_file, fs, audio_data)

            # Print message that recording has stopped
            print("Recording stopped.")

            # Transcribe the recorded audio
            transcription = model.transcribe([output_file])
            print(transcription)

            # Delete the output file
            os.remove(output_file)

            # Stop the listener and exit the program
            listener.stop()
            return False

init()