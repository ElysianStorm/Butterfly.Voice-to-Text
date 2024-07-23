# Butterfly 🦋: Voice to Text 

Welcome to Butterfly: Voice to Text, a comprehensive repository for converting audio to text using three state-of-the-art models: Whisper AI, Deep Speech, and VOSK. 

This repository provides the tools and codebase necessary to implement each model separately and understand the underlying processes involved in voice-to-text conversion. 

This project was a long kept to-do list, that I wanted to accomplish. Fascinated with "Jarvis", it was always a dream to be able to process audio directly into commands, with Butterfly the first step of the process is complete. I wish to share my experience in this domain for all who wish to explore this.

## Table of Contents
- [Project Overview](#project-overview)
- [Understanding VAD](#understanding-vad)
- [Implementing VAD from Scratch](#implementing-vad-from-scratch)
- [Model Selection](#model-selection)
- [Integrating Models with Software](#integrating-models-with-software)
- [Understanding Model Output](#understanding-model-output)
- [Future Scope](#future-scope)
- [Summary](#summary)
- [References](#references)

## Project Overview

Butterfly: Voice to Text is designed to facilitate the conversion of audio to text using three different AI models. Each model brings unique strengths to the table, providing flexibility in choosing the best tool for your specific needs. The repository includes code snippets, examples, and documentation to help you get started quickly.

## Understanding VAD

Voice Activity Detection (VAD) is a crucial component in voice-to-text systems. VAD identifies segments of audio that contain speech, allowing the system to focus on processing relevant parts of the audio stream. Both Whisper AI and VOSK utilize VAD to enhance their accuracy and efficiency. There are ton of research going around that dive extensively into this topic. As part of this project, I did study many of these papers, get meaningful insights from them and even implemented a rudimentary VAD from scratch. However, since we have already have more robut VAD's out there, it is better to work with them. I will briefly touch upon the VAD building from scratch within this readme in the section Implementing VAD from Scratch.

### Whisper AI VAD
Whisper AI uses an advanced VAD mechanism that leverages deep learning to accurately detect speech segments. This VAD system is integrated within Whisper's pipeline, ensuring that only significant audio frames are processed.

### VOSK VAD
VOSK employs a robust VAD system based on Kaldi's neural network models. VOSK's VAD efficiently filters out non-speech segments, optimizing the performance of the transcription process.

## Implementing VAD from Scratch

Implementing VAD involves several steps:

1. **Pre-processing Audio**: Convert audio to a suitable format and sample rate.
    ```python
    import pyaudio
    import wave

    def preprocess_audio(file_path):
        chunk = 1024
        wf = wave.open(file_path, 'rb')
        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(chunk)

        while data:
            stream.write(data)
            data = wf.readframes(chunk)

        stream.stop_stream()
        stream.close()
        p.terminate()
    ```

2. **Feature Extraction**: Extract features like Mel Frequency Cepstral Coefficients (MFCCs) to identify speech characteristics.
    ```python
    import numpy as np
    import python_speech_features as psf

    def extract_features(audio_data, sample_rate):
        mfcc_features = psf.mfcc(audio_data, samplerate=sample_rate, numcep=13)
        return mfcc_features
    ```

3. **Classification**: Use a classifier (e.g., neural networks) to distinguish between speech and non-speech segments.
    ```python
    from sklearn.neural_network import MLPClassifier

    def classify_segments(features):
        clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
        clf.fit(features, labels)  # `labels` should be predefined for supervised learning
        predictions = clf.predict(features)
        return predictions
    ```

4. **Post-processing**: Smooth the output to reduce false positives and improve detection accuracy.
    ```python
    def smooth_output(predictions, window_size=5):
        smoothed_predictions = np.convolve(predictions, np.ones(window_size)/window_size, mode='valid')
        return smoothed_predictions
    ```

## Model Selection

Choosing the right model depends on several factors:

- **Number of Parameters**: Models with more parameters generally offer higher accuracy but require more computational resources.
- **Training Data**: Models trained on diverse and extensive datasets perform better in real-world scenarios.
- **Robustness**: The ability of the model to handle noisy environments and various accents.

### Whisper AI
- **Parameters**: 1.5 billion
- **Training Data**: Diverse dataset including various accents and noise levels
- **Robustness**: High

### Deep Speech
- **Parameters**: 50 million
- **Training Data**: Large corpus of labeled speech data
- **Robustness**: Moderate

### VOSK
- **Parameters**: 25 million
- **Training Data**: Extensive data from multiple languages
- **Robustness**: High

## Integrating Models with Software

### Whisper AI Integration

1. **Installation**: Follow the setup instructions in the repository.
    ```bash
    pip install whisper
    ```

2. **API Integration**: Use the provided APIs to integrate Whisper AI into your application.
    ```python
    import whisper

    model = whisper.load_model("base")
    result = model.transcribe("audio.mp3")
    print(result["text"])
    ```

3. **Real-Time Processing**: Implement real-time audio capture and processing to achieve low-latency transcription.
    ```python
    import whisper
    import pyaudio

    model = whisper.load_model("base")

    def real_time_transcription():
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

        while True:
            data = stream.read(1024)
            result = model.transcribe(data)
            print(result["text"])

        stream.stop_stream()
        stream.close()
        p.terminate()
    ```

### Deep Speech Integration

1. **Installation**: Set up the environment as per the repository instructions.
    ```bash
    pip install deepspeech
    ```

2. **API Integration**: Utilize Deep Speech's API for transcription tasks.
    ```python
    from deepspeech import Model
    import wave

    model_file_path = 'deepspeech-0.9.3-models.pbmm'
    scorer_file_path = 'deepspeech-0.9.3-models.scorer'

    ds = Model(model_file_path)
    ds.enableExternalScorer(scorer_file_path)

    def transcribe_audio(file_path):
        with wave.open(file_path, 'rb') as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
            result = ds.stt(audio)
        return result

    print(transcribe_audio('audio.wav'))
    ```

3. **Output Handling**: Process and store the transcription output in desired formats.
    ```python
    def save_transcription(text, file_path):
        with open(file_path, 'w') as f:
            f.write(text)

    text = transcribe_audio('audio.wav')
    save_transcription(text, 'transcription.txt')
    ```

### VOSK Integration

1. **Installation**: Install VOSK as guided in the repository.
    ```bash
    pip install vosk
    ```

2. **API Integration**: Use VOSK's API to transcribe audio streams.
    ```python
    from vosk import Model, KaldiRecognizer
    import wave

    model = Model("model")
    wf = wave.open("audio.wav", "rb")
    rec = KaldiRecognizer(model, wf.getframerate())

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
        else:
            result = rec.PartialResult()
    
    print(result)
    ```

3. **Real-Time and Batch Processing**: Handle both real-time and batch audio processing efficiently.
    ```python
    import pyaudio
    from vosk import Model, KaldiRecognizer

    model = Model("model")
    rec = KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()

    def real_time_vosk():
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()

        while True:
            data = stream.read(4000)
            if rec.AcceptWaveform(data):
                result = rec.Result()
                print(result)

        stream.stop_stream()
        stream.close()
        p.terminate()

    real_time_vosk()
    ```

## Understanding Model Output

### Whisper AI
- **Temperature**: Controls the randomness of predictions.
- **Correctness**: Measure of transcription accuracy.
- **Probability Distribution**: Distribution of possible transcriptions.

    ```python
    result = model.transcribe("audio.mp3", temperature=0.5)
    print(result["text"])
    print(result["probability_distribution"])
    ```

### Deep Speech
- **Confidence Scores**: Likelihood of the transcription being correct.
- **Word Timing**: Timestamps for each word in the transcription.

    ```python
    from deepspeech import Model
    import wave

    def transcribe_audio_with_confidence(file_path):
        with wave.open(file_path, 'rb') as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
            result = ds.sttWithMetadata(audio)
        
        for transcript in result.transcripts:
            for token in transcript.tokens:
                print(f"{token.text} - Confidence: {token.confidence}")

    transcribe_audio_with_confidence('audio.wav')
    ```

### VOSK
- **Alternatives**: Multiple transcription possibilities.
- **Word Confidence**: Confidence level for each word.

    ```python
    from vosk import Model, KaldiRecognizer
    import wave

    model = Model("model")
    wf = wave.open("audio.wav", "rb")
    rec = KaldiRecognizer(model, wf.getframerate())

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            print(result)

    ```

## Future Scope

The future development of Butterfly: Voice to Text includes integrating these models with GPTs. The repository already contains code for a pipeline that sends the transcribed text to the Chat GPT API and processes the output.

### Future Enhancements
- **Improved VAD**: Further refinement of VAD for better accuracy.
- **Multilingual Support**: Expand support for more languages.
- **Enhanced Integration**: Seamless integration with more applications and APIs.

    ```python
    import openai

    openai.api_key = "your-api-key"

    def query_chat_gpt(transcribed_text):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=transcribed_text,
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].text

    transcribed_text = transcribe_audio('audio.wav')
    response = query_chat_gpt(transcribed_text)
    print(response)
    ```

## Summary

Butterfly: Voice to Text is a versatile and powerful tool for converting audio to text using Whisper AI, Deep Speech, and VOSK. By understanding VAD, implementing the right model, and integrating it effectively, you can achieve high-quality transcriptions for various applications.

## References

- [Whisper AI Documentation](https://github.com/openai/whisper)
- [Deep Speech Documentation](https://github.com/mozilla/DeepSpeech)
- [VOSK Documentation](https://github.com/alphacep/vosk-api)
- [Chat GPT API](https://beta.openai.com/docs/api-reference/chat)
