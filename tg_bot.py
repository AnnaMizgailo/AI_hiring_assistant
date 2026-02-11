import os
import asyncio
from typing import Any, Dict, Optional
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

async def api_create_application(job_id: str, telegram_user_id: int, telegram_username: Optional[str]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{API_BASE}/api/applications", json={"job_id": job_id, "telegram_user_id": telegram_user_id, "telegram_username": telegram_username})
        r.raise_for_status()
        return r.json()

async def api_list_jobs() -> list[dict]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{API_BASE}/api/jobs")
        r.raise_for_status()
        return r.json()

async def run_bot() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is empty")
    bot = Bot(BOT_TOKEN)
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def start(m: Message):
        jobs = await api_list_jobs()
        kb = InlineKeyboardBuilder()
        for j in jobs:
            kb.button(text=f'{j["title"]} (порог {j["threshold_score"]})', callback_data=f'job:{j["id"]}')
        await m.answer("Выберите вакансию:", reply_markup=kb.as_markup())

    @dp.callback_query(F.data.startswith("job:"))
    async def job_selected(c: CallbackQuery):
        job_id = c.data.split("job:", 1)[1]
        res = await api_create_application(job_id, c.from_user.id, c.from_user.username)
        await c.message.answer(f'Отклик создан. ID: {res["application_id"]}. Теперь пришлите резюме файлом.')
        await c.answer()

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(run_bot())
