# -*- ecoding: utf-8 -*-
# @ModuleName: baidu+robot
# @Function: 
# @Author: qy
# @Time: 2022/1/24 16:47
import speech_recognition as sr
from aip import AipSpeech
import requests
import json

import pygame

#录音
def rec(rate=16000):
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=rate) as source:
        print("please say something")
        audio = r.listen(source)

    with open("recording.wav", "wb") as f:
        f.write(audio.get_wav_data())

    return 1

#语音识别
APP_ID = '25553224'
API_KEY = 'R0eKeKmCprTP0PzTDK69m8uW'
SECRET_KEY = '2BelDodR7AZqG9TnVmk7GgjUMNfql1eP'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
def listen():
    with open('recording.wav', 'rb') as f:
        audio_data = f.read()

    results = client.asr(audio_data, 'wav', 16000, {
        'dev_pid': 1536,
        # dev_pid    语言    模型    是否有标点    备注
        #     1536    普通话(支持简单的英文识别)    搜索模型    无标点    仅采样率16000支持自定义词库
        # 1537    普通话(纯中文识别)    输入法模型    可以有标点    不支持自定义词库
        #     1736    英语    搜索模型    无标点    不支持自定义词库
        # 1737    英语    输入法模型    可以有标点    不支持自定义词库
        #     1636    粤语    搜索模型    无标点    不支持自定义词库
        # 1637    粤语    输入法模型    可以有标点    不支持自定义词库
        #     1836    四川话    搜索模型    无标点    不支持自定义词库
        # 1837    四川话    输入法模型    可以有标点    不支持自定义词库
    })
    if 'result' in results:
        print("you said: " + results['result'][0])
        return results['result'][0]
    else:
        print("出现错误，错误代码：" , results['err_no'])

#调用图灵机器人
TURING_KEY = "3519eaad5a5142fbbfd81838a53f1827"
URL = "http://openapi.tuling123.com/openapi/api/v2"
HEADERS = {'Content-Type': 'application/json;charset=UTF-8'}
def robot(text=""):
    data = {
        "reqType": 0,
        "perception": {
            "inputText": {
                "text": ""
            },
            "selfInfo": {
                "location": {
                    "city": "北京",
                    "street": "海淀区"
                }
            }
        },
        "userInfo": {
            "apiKey": 'abea0643e5f84cd9884e5caf9b46eb55',
            "userId": "123"
        }
    }

    data["perception"]["inputText"]["text"] = text
    response = requests.request("post", URL, json=data, headers=HEADERS)
    response_dict = json.loads(response.text)

    result = response_dict["results"][0]["values"]["text"]
    print("the AI said: " + result)
    return result





#语音合成
def speak(text=""):
    result = client.synthesis(text, 'zh', 1, {
        'spd': 4,
        'vol': 5,
        'per': 4,
    })

    if not isinstance(result, dict):
        with open('audio.mp3', 'wb') as f:
            f.write(result)

#播放音频
def play():
    pygame.mixer.init()
    pygame.mixer.music.load("audio.mp3")
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass
    pygame.mixer.music.unload()

if __name__ == "__main__":
    while True:
        rec()  # 保存录音文件：recording.wav
        text = listen()  # 自动打开录音文件recording.wav进行识别,返回 识别的文字存到text
        if '结束程序' in text:  #这里我设置了一个结束语，说“结束程序”的时候就结束，你也可以改掉
            break
        text_1 = robot(text)  # 将text中的文字发送给机器人，返回机器人的回复存到text_1
        speak(text_1)  # 将text_1中机器人的回复用语音输出，保存为audio.mp3文件
        play() #播放audio.mp3文件
