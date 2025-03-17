import os
from dotenv import load_dotenv
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk

env_path = os.path.dirname(os.path.dirname(__file__)) + os.path.sep + 'secrets.env'
load_dotenv(dotenv_path=env_path)
speechKey = os.getenv("SPEECH_KEY")
speechRegion = os.getenv("SPEECH_REGION")

print("Speech Key:: " + speechKey)
print("Speech Region:: " + speechRegion)

def recognize_from_microphone():
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=speechKey, region=speechRegion)
    speech_config.speech_recognition_language="en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        return speech_recognition_result.text

# print(recognize_from_microphone())