# The AudioTranscriber class provides a convenient interface for transcribing audio input using a loaded speech recognition model. 
# It allows for starting and stopping the audio recording using keypress events and provides a callback function 
# for processing the transcription result. The class uses the sounddevice, numpy, scipy.io.wavfile, and 
# pynput packages for audio recording and keypress event handling. 
# It also uses the huggingsound package for speech recognition. 
# 
# Overall, the AudioTranscriber class provides a useful tool for audio transcription in a variety of applications, 
# such as voice assistants or speech-to-text software.
import os
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pynput import keyboard
from huggingsound import SpeechRecognitionModel
import signal
import sys

class AudioTranscriber:
    def __init__(self, speech_recognition_model, transcription_callback=None):
        self.fs = 44100
        self.recording = []
        self.is_recording = False
        self.stream = None
        self.model = SpeechRecognitionModel(speech_recognition_model)
        self.listener = None
        self._initialize_signal_handler()
        self.transcription_callback = transcription_callback

    # Initializes the signal handler to gracefully exit the program on SIGINT.
    def _initialize_signal_handler(self):
        signal.signal(signal.SIGINT, self._signal_handler)

    # A signal handler function that is called when a SIGINT signal is received. It stops the audio recording and exits the program.
    def _signal_handler(self, sig, frame):
        print("\nClosing AudioTranscriber...")
        self.stop_recording()
        sys.exit(0)

    # Creates an audio input stream with a specified sample rate and number of channels.
    def create_stream(self):
        return sd.InputStream(samplerate=self.fs, channels=1, callback=self.audio_callback)

    # Transcribes an audio file using the loaded speech recognition model.
    def transcribe(self, audio_file):
        transcriptions = self.model.transcribe([audio_file])
        return transcriptions[0]["transcription"]

    # An audio callback function that is called when the input stream is active. 
    # It appends the incoming audio data to the recording list if is_recording is True.
    def audio_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.recording.append(indata.copy())

    # A keypress event handler that starts or stops the audio recording depending on the key pressed. 
    # It also stops the listener if the "esc" key is pressed.
    def on_press(self, key):
        if key == keyboard.Key.space:
            if not self.is_recording:
                print("Recording started...")
                self.is_recording = True
                self.stream = self.create_stream()
                self.stream.start()
            else:
                self.stop_recording()
        elif key == keyboard.Key.esc:
            self.stop_recording()
            return False

    # Stops the audio recording and returns the transcription of the recorded audio. 
    # It saves the recorded audio to a file and deletes the file after transcription. 
    # It also clears the recording data
    def stop_recording(self):
        transcription = None
        if self.is_recording:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            audio_data = np.concatenate(self.recording, axis=0)
            output_file = 'output.wav'
            write(output_file, self.fs, audio_data)
            print("Recording stopped.")
            transcription = self.transcribe(output_file)
            os.remove(output_file)
            if self.transcription_callback:
                self.transcription_callback(transcription)

            # Clear the recording data
            self.recording = []

        # Stop the listener: we need to do this because the listener is running in a separate thread 
        # and it would be difficult to get the return value in the callback function
        if self.listener:
            self.listener.stop()
        return transcription

    # Starts the listener for keypress events.
    def start(self):
        with keyboard.Listener(on_press=self.on_press) as self.listener:
            self.listener.join()