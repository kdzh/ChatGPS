# Смотрим на баланс: https://console.yandex.cloud/cloud/b1gg4ap6lni5ie5gno5e
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import requests
from datetime import datetime
import json
import subprocess
import os
from typing import Any, Generator, Optional, List

from streamlit_mic_recorder import speech_to_text
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.schema import Generation
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

import streamlit as st


from rag_methods import (
    SOURCE_DIR,
    DATA_DIR,
    initialize_vector_db,
)


USER_NAME = 'Пользователь'
ASSISTANT_NAME = 'Ассистент'

env = os.environ.copy()
env["IAM_TOKEN"] = (
    subprocess
    .run("export IAM_TOKEN=`yc iam create-token`; echo $IAM_TOKEN", shell=True, capture_output=True, check=True)
    .stdout
    .decode('utf-8')
)

# YANDEX_TOKEN = env["IAM_TOKEN"]


class LocalLLM(LLM):
    endpoint_url: str
    model_name: str

    headers: str = {'Content-Type': 'application/json'}
    last_answer: str = ''


    def _call(self, prompt: str, stop: list[str] | None = None) -> str:
        data = {
            'model': self.model_name,
            'messages': (
                    [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                    + [{'role': 'user', 'content': prompt}]
            ),
            'stream': False
        }

        # print('0:', prompt)
        # print('1:', [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]])
        # print('2:', json.dumps(data))
        # print('3:', data)
        # print('4:', self.headers)
        # print('5:', self.endpoint_url)

        response = requests.post(self.endpoint_url, headers=self.headers, data=json.dumps(data))
        assert response.status_code == 200

        # print(response.json())

        reply = response.json()["message"]["content"]
        return reply


    def _stream(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None) -> Generator[str, Any, None]:
        """Streaming call. Yields tokens as they arrive."""
        self.last_answer = ''
        data = {
            'model': self.model_name,
            # "messages": [{"role": "user", "content": prompt}],
            'messages': (
                    [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                    + [{'role': 'user', 'content': prompt}]
            ),
            'stream': True
        }
        with requests.post(
            self.endpoint_url,
            headers=self.headers,
            data=json.dumps(data),
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = line.decode("utf-8")
                    token = self._parse_token(data)
                    self.last_answer += token
                    yield token


    def _parse_token(self, line: str) -> str:
        """Parse the streamed line to extract the token text."""
        obj = json.loads(line)
        return obj["message"]['content']


    @property
    def _llm_type(self) -> str:
        return "local-llm"


class YandexLLM(LLM):
    temperature: float = 0.1
    last_answer: str = ''
    max_tokens: int = 2000

    def _call(self, prompt: str, stop: list[str] | None = None) -> str:
        prompt_dict = {
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt",
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": "2000",
                "reasoningOptions": {
                    "mode": "DISABLED"
                }
            },
            "messages": (
                    [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                    + [{'role': 'user', 'content': prompt}]
            ),
        }

        with open('yandex_prompt.json', 'w', encoding='utf8') as json_file:
            json.dump(prompt_dict, json_file, ensure_ascii=False)

        request_code = (
            """curl \
                  --request POST \
                  --header "Content-Type: application/json" \
                  --header "Authorization: Bearer ${IAM_TOKEN}" \
                  --data "@yandex_prompt.json" \
                  "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
            """
        )

        result = subprocess.run(
            request_code,
            shell=True,
            capture_output=True,
            text=True,
            env=env
        )

        print('result:', result)

        reply = json.loads(result.stdout)['result']['alternatives'][0]['message']['text']
        return reply


    def _stream(self, prompt: str, model_name = 'yandexgpt') -> Generator[str, Any, None]:
        self.last_answer = ''

        messages = (
                [{"role": m["role"], "text": m["content"]} for m in st.session_state.messages[:-1]]
                + [{'role': 'user', 'text': prompt}]
        )

        api_base = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {YANDEX_TOKEN}",
        }

        payload = {
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/{model_name}",
            "completionOptions": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
            },
            "messages": messages,
        }

        with requests.post(
                api_base,
                headers=headers,
                json=payload,
                stream=True,
                timeout=60
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = line.decode("utf-8")
                    token = self._parse_token(data)
                    truncated_token = token[len(self.last_answer):]
                    self.last_answer += truncated_token
                    yield truncated_token


        # text = ""
        # for line in response.iter_lines():
        #     if line:
        #         data = json.loads(line.decode("utf-8"))
        #         data = data["result"]
        #         top_alternative = data["alternatives"][0]
        #         text = top_alternative["message"]["text"]
        #
        #         print(text, end='', flush=True)
        #         # yield {"text": text, "error_code": 0}
        #
        #         status = top_alternative["status"]
        #         if status in (
        #                 "ALTERNATIVE_STATUS_FINAL",
        #                 "ALTERNATIVE_STATUS_TRUNCATED_FINAL",
        #         ):
        #             break

    @property
    def _llm_type(self) -> str:
        return "local-llm"


    def _parse_token(self, line: str) -> str:
        """Parse the streamed line to extract the token text."""
        obj = json.loads(line)
        return obj["result"]['alternatives'][0]["message"]["text"]





initial_prompt_template = """Отвечай по-русски. Если не знаешь ответа, просто скажи, что ты не знаешь.

Ниже предоставлены части документов (в разеделе "Контекст:"), которые могут быть использованы как контекст, чтобы ответить на вопрос.
Вопросы могут быть такие, что контекст не будет использован.

Если ты будешь использовать контекст для ответа, то последовательно сделай:
1) Приведи полностью ответ, используя контекст.
В ответе должны быть ссылки на источники в контексте по их номерам (например, [1], [2] и т.д.).
2) Пропусти строчку и процитируй полностью источник (перед источниками должны стоять те же номера).
Цитирование источника должно быть следующего формата. Вначале указываем его номер (например, "[1]").
Затем в скобках указываем название источника (значение из колонки "Источник") - название файла, название таблицы, номер строки из таблицы.

Затем пропускаем строчку и полностью процитируй сам источник (значения в остальных колонках). Оставляем знаки новой строки в источнике.

Приведенные источники должны быть приведены в порядке их использования в ответе.
Но поскольку номера этих источников (например, "[3]") могут таким образом идти в произвольном порядке, то перенумеруй их начиная с [1].
Перенумерованный вариант должен быть как ссылках в ответе, так и в номерах при приведении источников.


Контекст:
{context}

Вопрос:
{question}

Если в вопросе нет самого вопроса, то просто поддержи беседу.

Ответ:"""


if "messages" not in st.session_state:
    st.session_state.messages = []






if st.sidebar.button("🧹 Очистить чат"):
    st.session_state.messages.clear()
    st.rerun()

chat_lines = []
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", datetime.now().strftime("%H:%M"))
    chat_lines.append(f"{timestamp}, {role}: {content}")
chat_content = "\n".join(chat_lines)


st.sidebar.download_button(
    label="📄➜ Экспортировать чат",
    data=chat_content,
    file_name="История_чата.txt",
    mime="text/plain"
)


def microphone_callback():
    if st.session_state.text_received_from_the_mic:
        print(st.session_state.text_received_from_the_mic)
        st.session_state.text_received_from_the_mic = None


with st.sidebar:
    st.session_state.text_received_from_the_mic = speech_to_text(
        language='ru',
        start_prompt="🎤 Вкл. запись",
        stop_prompt="🎤 Выкл. запись",
        just_once=True,
        use_container_width=False,
        callback=microphone_callback,
        key='STT'
    )






st.sidebar.title("⚙️ Настройки")

st.sidebar.subheader('Параметры модели:')

MODEL_OPTIONS_DICT = {
    'YandexGPT': 'yandexgpt',
    'YandexGPT-lite': 'yandexgpt-lite',
    'Llama 3.1 8B': 'llama-lite',
    'Llama 3.3 70B': 'llama',
    'Gemma 3n E4B': 'gemma3n:e4b',
}
LOCAL_MODELS = ['Gemma 3n E4B']
selected_model_name = st.sidebar.selectbox("Выберете модель", MODEL_OPTIONS_DICT.keys())


YANDEX_FOLDER_ID = st.sidebar.text_input("Yandex ID облачного сервиса:", "b1gfbnbrsndktci2srd6")
YANDEX_DEFAULT_TOKEN = 't1.9euelZrPi4_IxouQj8iPyJmPkIzNze3rnpWaxpCPxpWXjpDOzM6dlceQxsrl9Pd7Pj07-e8UEA7H3fT3O206O_nvFBAOx83n9euelZqJzsyLjZvHmsqXmsaOipaJyO_8xeuelZqJzsyLjZvHmsqXmsaOipaJyA.pFo8Cp4MYQR3iSwP0Lj0e8VdbJT8d0z_sHBC05igwJTs908ciB6EQYBf7dg_abR5U863Ri8n6jFe3TG051NgCA'
YANDEX_TOKEN = st.sidebar.text_input("Yandex IAM-токен:", YANDEX_DEFAULT_TOKEN)


temperature = st.sidebar.slider('Температура:', 0.0, 1.0, 0.1)
max_tokens = st.sidebar.number_input("Максимальное кол-во токенов в ответе:", min_value=1, max_value=10000, value=2000, step=1)


# prompt_template = st.sidebar.text_input('Шаблон промпта:', initial_prompt_template)
prompt_template = initial_prompt_template

# st.sidebar.divider()

st.sidebar.subheader("Параметры поисковика:")
search_type = st.sidebar.selectbox('Тип поиска:', ('mmr', 'similarity', 'similarity_score_threshold'),)
score_threshold = st.sidebar.slider('Пороговое значение оценки (только если тип поиска similarity_score_threshold):', 0.0, 1.0, 0.5)
top_k_limit = st.sidebar.number_input(
    "Лимит на кол-во возвращаемых документов",
    min_value=1,
    max_value=100,
    value=None,
    placeholder="Укажите число...",
    step=1
)

st.sidebar.subheader('Параметры векторной базы:')

chunk_size = st.sidebar.number_input("Размер фрагмента:", min_value=100, max_value=10000, value=2000, step=1)
chunk_overlap = st.sidebar.number_input("Прекрытие фрагментов:", min_value=1, max_value=1000, value=500, step=1)


EMBEDDING_MODELS = [
    "sentence-transformers/LaBSE",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "DeepPavlov/rubert-base-cased-sentence"
]

embedding_model = st.sidebar.selectbox('Модель эмбеддингов:', tuple(EMBEDDING_MODELS),)
db_type = st.sidebar.selectbox("Тип базы:", ("FAISS", "Chroma"))

st.session_state.vector_db = initialize_vector_db(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    embedding_model=embedding_model,
    db_type=db_type
)




# st.sidebar.markdown("---")
# st.sidebar.markdown("**Демонстрационная модель RAG**")


st.markdown("""
    <style>
        .chat-title {
            font-size: 2.3rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 1.5rem;
            color: #4A90E2;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            max-height: 70vh;
            overflow-y: auto;
            padding: 10px;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        }
        .chat-bubble-user {
            align-self: flex-end;
            background-color: #DCF8C6;
            color: #000;
            padding: 12px;
            border-radius: 12px;
            margin: 5px 0;
            max-width: 80%;
        }
        .chat-bubble-assistant {
            align-self: flex-start;
            background-color: #F1F0F0;
            color: #000;
            padding: 12px;
            border-radius: 12px;
            margin: 5px 0;
            max-width: 80%;
        }
        .timestamp {
            font-size: 0.7rem;
            color: #888;
            margin: 0 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-title">💬 SigmaDevs - модель RAG</div>', unsafe_allow_html=True)




def get_chat_replica(role, content, timestamp):
    bubble_class = "chat-bubble-user" if role == 'user' else "chat-bubble-assistant"
    avatar = "🧑‍💻" if role == 'user' else "🤖"

    return (
        f'<div class="{bubble_class}">'
        f'<strong>{avatar} {(USER_NAME if role == 'user' else (ASSISTANT_NAME if role == 'assistant' else None)).capitalize()}:</strong><br>{content}'
        f'<div class="timestamp">{timestamp}</div>'
        f'</div>'
    )


if st.session_state.messages:
    # st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        timestamp = message.get("timestamp", datetime.now().strftime("%H:%M"))

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

        # st.markdown(
        #     get_chat_replica(role, content, timestamp),
        #     unsafe_allow_html=True
        # )

    # st.markdown('</div>', unsafe_allow_html=True)


if prompt := st.chat_input("Начните печатать сюда...") or st.session_state.text_received_from_the_mic:
    if st.session_state.text_received_from_the_mic:
        prompt = st.session_state.text_received_from_the_mic
        st.session_state.text_received_from_the_mic = None

    now = datetime.now().strftime("%H:%M")

    # st.markdown(
    #     get_chat_replica(role='user', content=prompt, timestamp=now),
    #     unsafe_allow_html=True
    # )

    with st.chat_message('user'):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": 'user',
        "content": prompt,
        "timestamp": now
    })

    if selected_model_name in LOCAL_MODELS:
        llm = LocalLLM(endpoint_url="http://localhost:11434/api/chat", model_name=MODEL_OPTIONS_DICT[selected_model_name])
    else:
        llm = YandexLLM(model=MODEL_OPTIONS_DICT[selected_model_name], temperature=temperature, max_tokens=max_tokens)

    retriever_search_kwargs = {}
    if search_type == 'similarity_score_threshold':
        retriever_search_kwargs["score_threshold"] = score_threshold
    if top_k_limit is not None:
        retriever_search_kwargs['k'] = top_k_limit
    retriever = st.session_state.vector_db.as_retriever(search_type=search_type, search_kwargs=retriever_search_kwargs)

    docs = retriever.invoke(prompt)

    numbered_chunks = []
    for i, doc in enumerate(docs):
        numbered_chunks.append(f"[{i + 1}] {doc.page_content}")

    context = "\n\n".join(numbered_chunks)
    # context = "\n\n".join([d.page_content for d in docs])

    custom_prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    full_prompt = custom_prompt_template.format(context=context, question=prompt)

    with st.chat_message("assistant"):
        st.write_stream(llm._stream(full_prompt))

    st.session_state.messages.append({
        "role": 'assistant',
        "content": llm.last_answer,
        "timestamp": datetime.now().strftime("%H:%M")
    })



    # Rerun to show the new messages
    st.rerun()
