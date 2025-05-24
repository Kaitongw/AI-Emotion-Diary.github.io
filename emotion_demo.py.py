import cv2
import numpy as np
import tensorflow as tf
import textwrap

# ✅ 模擬日記字典（Demo用，不需GPT）
fake_diaries = {
    "Happy": "I felt cheerful today. Everything seemed brighter and more joyful.",
    "Sad": "I felt a little down today, but I believe tomorrow will be better.",
    "Angry": "I was frustrated by some things today, but I’ll try to stay calm."
}

# 模擬 GPT 生成函式
def generate_demo_diary(emotion):
    diary = fake_diaries.get(emotion, "今天過得平平淡淡，但我仍心懷希望。")
    print(f"[DEMO 模式] 偵測到 {emotion}：{diary}")
    with open("emotion_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{emotion}] {diary}\n")
    return diary

# 匯入 TFLite 模型
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ['Happy', 'Sad', 'Angry']
cap = cv2.VideoCapture(0)

last_emotion = ""
diary_text = ""

# 圖像預處理
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img = img / 255.0
    return img

# 主程式執行
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.flip(frame, 1)
    input_data = preprocess(img)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    label = classes[np.argmax(output_data)]

    # 如果表情改變，就更新日記
    if label != last_emotion:
        print(f"⚠️ 偵測到情緒變化：{last_emotion} → {label}")
        last_emotion = label
        diary_text = generate_demo_diary(label)

    # 顯示辨識情緒與假日記文字在畫面
    cv2.putText(img, f"Emotion: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    y0 = 70
    wrapped_lines = textwrap.wrap(diary_text, width=40)  # 每 40 字自動斷行
    for i, line in enumerate(wrapped_lines):
        cv2.putText(img, line, (10, y0 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


