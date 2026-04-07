import os
import asyncio
import logging
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart

# Импортируем скомпилированный граф из нашего модуля agent
from agent import graph_app

# Настройка логирования для отслеживания возможных ошибок
logging.basicConfig(level=logging.INFO)

# Загрузка переменных окружения
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не найден в файле .env")

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    """
    Хэндлер на команду /start.
    Передает базовое приветствие в граф, чтобы LLM сама инициировала диалог
    согласно системному промпту.
    """
    thread_id = str(message.from_user.id)
    config = {"configurable": {"thread_id": thread_id}}

    # Отправляем системный триггер агенту
    initial_input = {"messages": [("user", "Привет! Я готов пройти опрос для оценки риска заболевания печени.")]}

    # Используем ainvoke для асинхронного вызова
    result = await graph_app.ainvoke(initial_input, config)

    # Извлекаем последний ответ от агента и отправляем пользователю
    agent_response = result["messages"][-1].content
    await message.answer(agent_response)


@dp.message(F.text)
async def handle_user_text(message: types.Message):
    """
    Универсальный хэндлер для всех текстовых сообщений.
    Работает без конечных автоматов (FSM) aiogram, всю логику ведет LangGraph.
    """
    thread_id = str(message.from_user.id)
    config = {"configurable": {"thread_id": thread_id}}

    user_input = {"messages": [("user", message.text)]}

    # Отправляем пользователю индикатор "печатает...", так как вызов модели может занять пару секунд
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        # Передаем сообщение пользователя в граф
        result = await graph_app.ainvoke(user_input, config)

        # Получаем итоговый ответ
        agent_response = result["messages"][-1].content

        await message.answer(agent_response)

    except Exception as e:
        logging.error(f"Ошибка при обработке сообщения графом: {e}")
        await message.answer("Произошла системная ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.")


async def main():
    """Точка входа для запуска поллинга."""
    logging.info("Запуск Telegram-бота...")
    # Удаляем вебхуки (если были) и запускаем поллинг
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Бот остановлен пользователем.")