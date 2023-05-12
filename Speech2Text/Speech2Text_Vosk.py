import subprocess
import json
from torch import channels_last
from vosk import Model, KaldiRecognizer
import time
import pyaudio
from queue import Queue
from threading import Thread
import keyboard

messages = Queue()
recordings = Queue()

CHANNELS = 1
FRAME_RATE = 16000
RECORD_SECONDS = 20
AUDIO_FORMAT = pyaudio.paInt16
SAMPLE_SIZE = 2

model = Model(model_name="vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(model, FRAME_RATE)
rec.SetWords(True)
    
def record_microphone(chunk=1024):
    p = pyaudio.PyAudio()

    stream = p.open(format=AUDIO_FORMAT,
                    channels=CHANNELS,
                    rate=FRAME_RATE,
                    input=True,
                    frames_per_buffer=chunk)

    frames = []

    while not messages.empty():
        data = stream.read(chunk)
        frames.append(data)
        if len(frames) >= (FRAME_RATE * RECORD_SECONDS) / chunk:
            recordings.put(frames.copy())
            frames = []

    stream.stop_stream()
    stream.close()
    p.terminate()
    
def speech_recognition():
    
    while not messages.empty():
        frames = recordings.get()
        
        rec.AcceptWaveform(b''.join(frames))
        result = rec.Result()
        text = json.loads(result)["text"]
        
        # cased = subprocess.check_output('python recasepunc/recasepunc.py predict recasepunc/checkpoint', shell=True, text=True, input=text)
        # output.append_stdout(cased)
        # time.sleep(1)

def start_recording():
    messages.put(True)
    record_microphone()
    print("Recording Complete")
    speech_recognition()
    print("Speech Recog Complete")

def stop_recording():
    messages.get()
    print("Stopped.")

def main():
    print("Hold SPACE To Record")
    keyboard.wait('space')
    if keyboard.is_pressed('space'):
        print('Recording...')

        while keyboard.is_pressed('space'):
            start_recording()
        
    stop_recording()

if __name__ == '__main__':
    main()