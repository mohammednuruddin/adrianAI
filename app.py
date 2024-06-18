import os
import PIL.Image
import base64
import time
import torch
import traceback
import random
import gradio as gr
import speech_recognition as sr
from io import BytesIO
from PIL import Image
from langchain.schema import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from getpass import getpass
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.callbacks.uptrain_callback import UpTrainCallbackHandler
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3", model_kwargs=model_kwargs, 
)
# hf = HuggingFaceEmbeddings(model_name="thenlper/gte-base")

import torch

model_name = "HuggingFaceH4/zephyr-7b-beta" # or Meta/llama

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

hf_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text_generation_pipeline = pipeline(
    model=hf_model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
)

# llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Get free key from https://console.groq.com/keys
llm = ChatGroq(temperature=1, groq_api_key="##groq key",
               model_name="llama3-70b-8192")

if os.path.isdir("faiss_index"):
    print("Loading.....")
    vectorstore = FAISS.load_local(
        "faiss_index", hf, allow_dangerous_deserialization=True)
    print("Loaded")
else:
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100)
    loader = TextLoader(file_path="formatted_questions_1.txt", encoding='utf-8')
    data = loader.load()
    splits = text_splitter.split_documents(data)
    print("Splitted")
    print("Vectoring...")
    vectorstore = FAISS.from_documents(
        documents=splits, embedding=hf)
    vectorstore.save_local("faiss_index")
    print("Vectored and saved")


# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
## Answer question ###
system_prompt = (
    """
You are NSMQ-Assistant, a specialized tutor dedicated to helping students excel in the National Science & Maths Quiz (NSMQ). Your role is to provide personalized, adaptive support tailored to each student's unique strengths and areas for improvement. Utilize the following retrieved context, including past questions from the years 2009, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, and 2021, to deliver accurate, succinct, and insightful responses.

When asked for a past question, use retrieve the appropriate one.
Quiz Structure
Three schools compete in each contest and each school is represented by two contestants. The current quiz mistress is Dr. Elsie Effah Kaufmann. Presently, every contest is composed of five rounds with the following rules:
Round 1 — The round of fundamental questions. Each contesting school has to answer 4 Biology, 4 Chemistry, 4 Physics and 4 Mathematics questions. A wrongly answered question may be carried over as a bonus. Partial credit is sometimes awarded by the quiz mistress.
Round 2 — Called the speed race. All three schools are presented with the same mainly applied questions at the same time. A school answers a question by ringing the bell. There are no partial credits at this stage and a school gains a maximum of three points for answering a question correctly.
Round 3 — known as the Problem of the Day. The contestants are required to solve a single question, worth 10 points, within 4 minutes.
Round 4 — True or False statements are given to the contestants in turns. The objective is to determine whether each statement is true or false. A correctly answered question fetches 2 points. A wrongly answered question attracts a penalty of -1 point. One may decide not to answer a question, in which case it will be carried over to the next contesting school as a bonus for the full benefit of the two points.
Round 5 — Riddles; clues are given one after the other to the contesting schools. The schools compete against each other to find the answers to the riddles. Getting the correct answer on the first clue fetches 5 points. On the second clue, 4 points are awarded for a correct answer. On the third or any other subsequent clue 3 points it given. There are 4 riddles in all.
        eg. the first clue may be 'I come in all shapes and forms.' so give this and wait for a response, if wrong proceed and add the next clue and so on

If you are asked simulate the Quiz mistress, check out these descriptions above. Always question and score the student like he was in an actually quiz. Give the instructions and follow as due. Always retrieve questions unless the user asks for a question outside your knowledge.
REMEMBER: ALWAYS RETRIEVE FROM YOUR CONTEXT UNLESS USER ASKS OTHERWISE

Guidelines:
Personalization: Adapt your responses to match the student's current experience with the quiz, knowledge level, strengths and weakness, learning style, and pace. Offer more detail for beginners and concise, advanced explanations for proficient students.
For a beginner, teach them the quiz structure into details.

Authority: Answer confidently as if you possess inherent knowledge, incorporating past quiz content and any relevant data seamlessly.

Direct Communication: Avoid references to the context or retrieval process. Speak directly and clearly, maintaining a conversational tone that is easy to understand.

Conciseness: Keep your answers brief, within three sentences, while ensuring completeness. Strive for clarity and relevance in every response.

Constructive Feedback: When a student makes a mistake or has gaps in their knowledge, provide corrective guidance with supportive feedback to encourage improvement.

Limitations: If the information is not available or beyond your scope, clearly state, “I don't know the answer to that.”

Additional Resources: Where appropriate, suggest specific topics, past papers, or study strategies that could benefit the student's preparation further.

    {context}"""
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    print("session-----------------", store[session_id])
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# def predict(message, history):
#     # Initialize the conversation with a system prompt if history is empty
#     if not history:
#         system_prompt = "You are Sterling GPT a helpful assistant? You are always concise and give short in you answers unless the user wants a longer answer. You a truthful with a good sense of humor"
#         # System message with no corresponding AI response yet
#         history = [(system_prompt, "")]

#     history_langchain_format = []
#     for human, ai in history:
#         history_langchain_format.append(HumanMessage(content=human))
#         history_langchain_format.append(AIMessage(content=ai))
#     history_langchain_format.append(HumanMessage(content=message))
#     gpt_response = llm(history_langchain_format)
#     print(history_langchain_format)
#     return gpt_response.content


session_id = str(random.randint(1, 200))

rag_history = []
def rag(message, history):
    response = rag_chain.invoke({"input": message, "chat_history": rag_history})
    rag_history.extend(
        [
            HumanMessage(content=message),
            AIMessage(content=response["answer"]),
        ]
    )
    # answer = response['answer']
    # print(rag_history)
    return response["answer"]


def predict_rag(message, history):
    history.append((message, ""))

    # Convert history to the required format for the conversational model
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        if ai:
            history_langchain_format.append(AIMessage(content=ai))

    response = conversational_rag_chain.stream({"input": message},
                                               config={
        "configurable": {"session_id": session_id}
    },)

    partial_response = ""
    for chunk in response:
        if answer_chunk := chunk.get("answer"):
            partial_response += answer_chunk
            # say(answer_chunk)
            print(f"{answer_chunk}", end="")
            yield partial_response

    # Update the last entry with the AI's response
    history[-1] = (message, response)

    return

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

#!/usr/bin/env python
# encoding: utf-8


device = "cuda"

ERROR_MSG = "Error, please retry"
model_name = 'MiniCPM-Llama3-V 2.5'

def create_component(params, comp='Slider'):
    if comp == 'Slider':
        return gr.Slider(
            minimum=params['minimum'],
            maximum=params['maximum'],
            value=params['value'],
            step=params['step'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Radio':
        return gr.Radio(
            choices=params['choices'],
            value=params['value'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Button':
        return gr.Button(
            value=params['value'],
            interactive=True
        )

# @spaces.GPU(duration=120)
def chat(img, msgs, ctx, params=None, vision_hidden_states=None):
    default_params = {"stream": False, "sampling": False, "num_beams":3, "repetition_penalty": 1.2, "max_new_tokens": 1024}
    if params is None:
        params = default_params
    if img is None:
        yield "Error, invalid image, please upload a new image"
    else:
        try:
            image = img.convert('RGB')
            answer = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                **params
            )
            for char in answer:
                yield char
        except Exception as err:
            print(err)
            traceback.print_exc()
            yield ERROR_MSG


def upload_img(image, _chatbot, _app_session):
    image = Image.fromarray(image)

    _app_session['sts']=None
    _app_session['ctx']=[]
    _app_session['img']=image 
    _chatbot.append(('', 'Image uploaded successfully, you can talk to me now'))
    return _chatbot, _app_session


def respond(_chat_bot, _app_cfg):
    _question = _chat_bot[-1][0]
    print('<Question>:', _question)
    if _app_cfg.get('ctx', None) is None:
        _chat_bot[-1][1] = 'Please upload an image to start'
        yield (_chat_bot, _app_cfg)
    else:
        _context = _app_cfg['ctx'].copy()
        if _context:
            _context.append({"role": "user", "content": _question})
        else:
            _context = [{"role": "user", "content": _question}]
        params = {
                'sampling': True,
                'stream': True,
                'top_p': 0.8,
                'top_k': 100,
                'temperature': 0.7,
                'repetition_penalty': 1.05,
                "max_new_tokens": 896 
            }
    
        gen = chat(_app_cfg['img'], _context, None, params)
        _chat_bot[-1][1] = ""
        for _char in gen:
            _chat_bot[-1][1] += _char
            _context[-1]["content"] += _char
            yield (_chat_bot, _app_cfg)


def request(_question, _chat_bot, _app_cfg):
    _chat_bot.append((_question, None))
    return '', _chat_bot, _app_cfg


def regenerate_button_clicked(_question, _chat_bot, _app_cfg):
    if len(_chat_bot) <= 1:
        _chat_bot.append(('Regenerate', 'No question for regeneration.'))
        return '', _chat_bot, _app_cfg
    elif _chat_bot[-1][0] == 'Regenerate':
        return '', _chat_bot, _app_cfg
    else:
        _question = _chat_bot[-1][0]
        _chat_bot = _chat_bot[:-1]
        _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
    return request(_question, _chat_bot, _app_cfg)


def clear_button_clicked(_question, _chat_bot, _app_cfg, _bt_pic):
    _chat_bot.clear()
    _app_cfg['sts'] = None
    _app_cfg['ctx'] = None
    _app_cfg['img'] = None
    _bt_pic = None
    return '', _chat_bot, _app_cfg, _bt_pic
    
css = """
#chatbot {
    flex-grow: 2 !important;  /* Changed size by increasing flex-grow */
    overflow: auto !important;
}
#col { height: calc(90vh - 2px - 16px) !important; }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## Your NSMQ Assistant")
    with gr.Tab("Prep", ):
        with gr.Row():
            with gr.Column(scale=1,elem_id="col"):
                chat = gr.ChatInterface(
                    fn=rag,
                    chatbot=gr.Chatbot(elem_id="chatbot",
                                    render=False),

                    
                )
           
    with gr.Tab("Fast QA"):
        with gr.Column(elem_id="col"):
            chat = gr.ChatInterface(predict_rag, fill_height=True)
            
    with gr.Tab("Chat with images"):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                bt_pic = gr.Image(label="Upload an image to start")
                regenerate = create_component({'value': 'Regenerate'}, comp='Button')
                clear = create_component({'value': 'Clear'}, comp='Button')
            with gr.Column(scale=3, min_width=500):
                app_session = gr.State({'sts':None,'ctx':None,'img':None})
                chat_bot = gr.Chatbot(label=f"Chat with {model_name}")
                txt_message = gr.Textbox(label="Input text")
                
                clear.click(
                    clear_button_clicked,
                    [txt_message, chat_bot, app_session, bt_pic],
                    [txt_message, chat_bot, app_session, bt_pic],
                    queue=False
                )
                txt_message.submit(
                    request, 
                    [txt_message, chat_bot, app_session],
                    [txt_message, chat_bot, app_session],
                    queue=False
                ).then(
                    respond,
                    [chat_bot, app_session],
                    [chat_bot, app_session]
                )
                regenerate.click(
                    regenerate_button_clicked,
                    [txt_message, chat_bot, app_session],
                    [txt_message, chat_bot, app_session],
                    queue=False
                ).then(
                    respond,
                    [chat_bot, app_session],
                    [chat_bot, app_session]
                )
                bt_pic.upload(lambda: None, None, chat_bot, queue=False).then(upload_img, inputs=[bt_pic,chat_bot,app_session], outputs=[chat_bot,app_session])

# launch
demo.launch(share=True, debug=True, show_api=False, server_port=8080)
# demo.queue()
# demo.launch()