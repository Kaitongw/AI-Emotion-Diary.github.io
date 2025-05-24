
# AI Emotion Recognition Diary (Demo Version)

This is a simple AI project that uses a webcam to detect facial emotions and automatically generates a short mood diary entry based on the detected emotion.

## 🧠 Project Overview

This Python-based demo uses a TensorFlow Lite model exported from [Teachable Machine](https://teachablemachine.withgoogle.com/) to recognize facial expressions, and simulates the diary generation with built-in English phrases for demonstration purposes (no API key required).

## 🎯 Features

- Real-time webcam emotion detection (Happy, Sad, Angry)
- Simulated mood diary output
- Diary log saved automatically to `emotion_log.txt`
- OpenCV-based visualization of emotion and diary
- **No internet or OpenAI API key required**

## 🖥️ How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/AI-Emotion-Diary.git
cd AI-Emotion-Diary
```

2. Install required packages

```bash
pip install opencv-python numpy tensorflow
```

3. Run the demo

```bash
python emotion_demo.py
```

4. Press `q` to quit the webcam view.

## 📂 Project Structure

```
.
├── emotion_demo.py         # Main demo script (no GPT required)
├── model.tflite            # Teachable Machine exported model
├── emotion_log.txt         # Output diary entries (auto-generated)
└── README.md               # Project introduction (this file)
```

## ✨ Sample Output

```
[DEMO 模式] 偵測到 Happy：I felt cheerful today. Everything seemed brighter and more joyful.
```

## 📌 Notes

- This is a demo version meant for showcasing the workflow.
- For real diary generation using ChatGPT, use `openai.ChatCompletion` and set up a valid API key.

---

© 2025 AI Emotion Project | Demo for Final Report Presentation
