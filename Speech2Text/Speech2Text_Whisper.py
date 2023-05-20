from os import write
import pyaudio
import wave
import whisper
import torch
import keyboard

# Initialize the device
device = "cuda" if torch.cuda.is_available() else "cpu"

class audioFileCreate(object):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample                                                                                                                                                                
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    seconds = 5
    filename = ".tempAudio.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    stream = p.open(format=sample_format,
                                channels=channels,
                                rate=fs,
                                frames_per_buffer=chunk,
                                input=True)
    
    frames = []  # Initialize array to store frames

    print('Hold SPACE to record')
    keyboard.wait('space')
    if keyboard.is_pressed('space'):
        print('Recording...')

        while keyboard.is_pressed('space'):
            data = stream.read(chunk)
            frames.append(data)

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("File created")

def generateTranscribedText(object):
    textFile = open("TranscribedText.txt", "w+")
    textFile.write(object["text"])
    textFile.close()
    print("Transcribed text file created.")


def main():
    audioFile = audioFileCreate()
    model = whisper.load_model("tiny")
    
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audioFile.filename)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)
    print(f"Detected language: {lang}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    # print(result.text)

    # translation for other languages
    print(f"Translated text from {lang} to en")
    output = model.transcribe(audioFile.filename, task = 'translate')
    print(output["text"])
    generateTranscribedText(output)
    
if __name__ == '__main__':
    main()