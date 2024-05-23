from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import speech_recognition as sr
from django.http import StreamingHttpResponse
from flask import Flask, render_template, request, redirect, url_for, flash, make_response, Response
from flask import Blueprint, render_template
from argon2 import PasswordHasher
from passlib.hash import sha256_crypt
import cv2
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
import numpy as np
import pyaudio
import wave
import speech_recognition as sr
import openai
import google.generativeai as genai


from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from django.contrib import messages
from .forms import SignUpForm, UsernamePasswordResetForm
from django.contrib.auth.forms import AuthenticationForm



genai.configure(api_key="AIzaSyDz_LTs21KNJJ0o_q4ZnmxwqumiDzPfbX0")

# Gemini setup
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 1024,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": ["hi"]
  },
  {
    "role": "model",
    "parts": ["Hello! 👋  How can I help you today?"]
  },
])


r = sr.Recognizer()
camera = cv2.VideoCapture(0)
ph = PasswordHasher()
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('Emotion_Detection.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprised']

def index(request):
    return render(request, 'index.html')

@login_required(login_url='/signup/')
def panel(request):
    perms = 'none'
    return render(request, "panel.html", {'perms': perms})

@csrf_exempt
def chat(request):
    text2 = "none"
    s = "none"
    text = "none"

    if request.method == 'POST':
        freq = 44100
        duration = 10

        recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
        sd.wait()
        write("recording0.wav", freq, recording)
        wv.write("recording1.wav", recording, freq, sampwidth=2)

        srtran = sr.AudioFile('recording1.wav')
        with srtran as source:
            audio = r.record(source)
        try:
            s = r.recognize_google(audio)
            print("Text: " + s)
        except Exception as e:
            print("Exception: " + str(e))

        completion = convo.send_message(f"What kind of emotion is this text expressing, say it in no more than one word from these emotions (Happy, Angry, Surprise, Sad, Fear, and Neutral): {s}\nAI:")
        text = convo.last.text

        completion2 = convo.send_message(f"You are a therapist. Don't give heading in reply with **, just act as therapist. Using this input from the user: {s}, and the emotion of the user from that text: {text}, say something which will help their mood and give recommendations to make their situation better. Ask related questions as a therapist\nAI:")
        text2 = convo.last.text
        print(text2)

    perms = 'none'
    return render(request, "chat.html", {'perms': perms, 'text2': text2, 'ftext': s, 'text': text})

def gen_frames():
     # Capture from the default camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_frame():
    if camera.isOpened():
        ret, frame = camera.read() 
        if ret:
            return(ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            return ret, None
    else:
        return None
    


def gen_frames():  
    while True:
        success, frame = camera.read()  # read the cap frame
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),10)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)


                    preds = classifier.predict(roi)[0]
                    #print("\nprediction = ",preds)
                    label=class_labels[preds.argmax()]
                    print(label)

                    
                    #print("\nprediction max = ",preds.argmax())
                    #print("\nlabel = ",label)
                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(40, 224, 129),5)
                else:
                    cv2.putText(frame,'Please make certain there is a face in front of the Camera.',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(40, 224, 129),3)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            


def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password1'])
            user.save()
            login(request, user)
            return redirect('login')
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('index')
            else:
                messages.error(request, 'Invalid username or password')
        else:
            messages.error(request, 'Invalid username or password')
    form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def password_reset_view(request):
    if request.method == 'POST':
        form = UsernamePasswordResetForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            new_password = form.cleaned_data.get('new_password')
            try:
                user = User.objects.get(username=username)
                user.password = make_password(new_password)
                user.save()
                messages.success(request, 'Password has been reset successfully.')
                return redirect('login')
            except User.DoesNotExist:
                messages.error(request, 'User with this username does not exist.')
    else:
        form = UsernamePasswordResetForm()
    return render(request, 'password_reset.html', {'form': form})