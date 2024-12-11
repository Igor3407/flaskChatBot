import streamlit as st
import torch
import random
import json
from transformers import pipeline
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import sentencepiece
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("milyausha2801/rubert-russian-qa-sberquad")
hf_model = AutoModelForQuestionAnswering.from_pretrained("milyausha2801/rubert-russian-qa-sberquad")

print("Модель загружена:", 'hf_model')
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

        if prob.item() > 0.4:
            tag_index = predicted.item()
            tag = tags[tag_index]

            for intent in intents:
                if intent['tag'] == tag:
                    return random.choice(intent['responses'])
        else:
            # Генерация ответа с использованием самого вопроса как контекста
            context = user_input
            inputs = tokenizer.encode_plus(
                user_input,
                context,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            outputs = hf_model(**inputs)

            answer_start = torch.argmax(outputs.start_logits).item()
            answer_end = torch.argmax(outputs.end_logits).item() + 1

            input_ids = inputs['input_ids'][0]

            # Проверка индексов
            if answer_start < 0 or answer_end > len(input_ids) or answer_start >= answer_end:
                return "Извините, я не смог найти ответ на ваш вопрос, но я постараюсь помочь!"

            # Преобразование токенов в строку
            tokens = tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
            answer = tokenizer.convert_tokens_to_string(tokens)

            return answer
    except Exception as e:
        return f"Ошибка: {str(e)}" ###f"задайте вопрос по другому"


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