import streamlit as st
import torch
import random
import json
from transformers import pipeline
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import streamlit as st
import torch
import random
import json
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Загрузка модели
@st.cache_resource
def load_model():
    FILE = "data.pth"
    data = torch.load(FILE, weights_only=True)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    model.eval()

    return model, all_words, tags


# Загрузка intents
@st.cache_data
def load_intents():
    with open('intents.json', 'r', encoding='utf-8') as file:
        return json.load(file)['intents']


# Инициализация модели Hugging Face для генерации ответов
# hf_model = pipeline("question-answering", model="AndrewChar/model-QA-5-epoch-RU")
# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("AndrewChar/model-QA-5-epoch-RU")
hf_model = AutoModelForQuestionAnswering.from_pretrained("AndrewChar/model-QA-5-epoch-RU", from_tf=True)


model, all_words, tags = load_model()
intents = load_intents()


# Функция для получения ответа от модели
def get_response(user_input):
    try:
        sentence = tokenize(user_input)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X_tensor = torch.from_numpy(X)

        output = model(X_tensor)
        prob, predicted = torch.max(output, dim=1)

        if prob.item() > 0.55:
            tag_index = predicted.item()
            tag = tags[tag_index]

            for intent in intents:
                if intent['tag'] == tag:
                    return random.choice(intent['responses'])
        else:
            # Генерация ответа с помощью модели Hugging Face
            response = hf_model(question=user_input, context="")  # Add context if needed
            return response['answer']
    except Exception as e:
        return f"Произошла ошибка: {str(e)}"


# Streamlit интерфейс
st.title("Чат-бот на основе ИИ")
st.write("Введите ваш вопрос:")

user_input = st.text_input("Ваш вопрос:")

if st.button("Получить ответ"):
    if user_input:
        with st.spinner("Обработка..."):
            response = get_response(user_input)
        st.write("Ответ:", response)
    else:
        st.write("Пожалуйста, введите вопрос.")