import streamlit as st
import torch
import random
import json
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Загрузка модели
FILE = "data.pth"
data = torch.load(FILE, weights_only=True)

# Извлечение параметров модели
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Загрузка intents из файла
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)['intents']

# Инициализация модели
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Функция для получения ответа от модели
def get_response(user_input):
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X_tensor = torch.from_numpy(X)

    output = model(X_tensor)
    prob, predicted = torch.max(output, dim=1)

    # Проверяем вероятность предсказания
    if prob.item() > 0.75:
        tag_index = predicted.item()

        if tag_index < len(tags):
            tag = tags[tag_index]
            print(f"Предсказанный тег: {tag}, Индекс: {tag_index}")

            # Находим соответствующий intent по тегу
            for intent in intents:
                if intent['tag'] == tag:
                    return random.choice(intent['responses'])  # Возвращаем случайный ответ из responses
    return "Извините, я не понял ваш вопрос."

# Streamlit интерфейс
st.title("Чат-бот на основе ИИ")
st.write("Введите ваш вопрос:")

user_input = st.text_input("Ваш вопрос:")

if st.button("Получить ответ"):
    if user_input:
        response = get_response(user_input)
        st.write("Ответ:", response)
    else:
        st.write("Пожалуйста, введите вопрос.")