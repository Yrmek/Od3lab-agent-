import os
import joblib
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Загружаем переменные окружения
load_dotenv()

# Чтение системного промпта из внешнего файла
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as file:
        SYSTEM_PROMPT = file.read()
except FileNotFoundError:
    raise FileNotFoundError("Файл system_prompt.txt не найден. Создайте его в корневой директории.")


# 1. Определение инструмента (Tool) для вызова ML-модели
@tool
def predict_liver_disease(
        age: float, gender: int, bmi: float, alcohol_consumption: float,
        smoking: int, genetic_risk: int, physical_activity: float,
        diabetes: int, hypertension: int, liver_function_test: float
) -> str:
    """
    Инструмент для предсказания вероятности заболевания печени.
    Принимает на вход 10 медицинских показателей пациента и возвращает вероятность риска в процентах.
    """
    try:
        model = joblib.load('random_forest_model.joblib')

        data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'BMI': bmi,
            'AlcoholConsumption': alcohol_consumption,
            'Smoking': smoking,
            'GeneticRisk': genetic_risk,
            'PhysicalActivity': physical_activity,
            'Diabetes': diabetes,
            'Hypertension': hypertension,
            'LiverFunctionTest': liver_function_test
        }])

        probability = model.predict_proba(data)[0][1]
        return f"Вероятность заболевания: {probability * 100:.2f}%"

    except Exception as e:
        return f"Произошла ошибка при вызове ML-модели: {str(e)}"


tools = [predict_liver_disease]

# 2. Инициализация LLM через OpenRouter API
# Укажите желаемую модель из каталога OpenRouter (например, meta-llama/llama-3.1-8b-instruct или openai/gpt-4o-mini)
llm = ChatOpenAI(
    model="google/gemini-2.5-flash-lite",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2
)
llm_with_tools = llm.bind_tools(tools)


# 3. Узлы графа
def call_model(state: MessagesState):
    """Узел вызова LLM"""
    messages = state["messages"]
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# 4. Сборка графа (LangGraph)
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

memory = MemorySaver()
graph_app = workflow.compile(checkpointer=memory)