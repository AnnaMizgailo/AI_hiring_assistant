import os
import asyncio
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder
import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
RECRUITER_ID = "demo-recruiter"

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is empty. Проверьте .env файл")

class CandidateFlow(StatesGroup):
    SELECTING_JOB      = State()
    WAITING_RESUME     = State()
    PROCESSING_RESUME  = State()
    ASK_SALARY         = State()
    ASK_WORK_FORMAT    = State()
    ASK_REASON         = State()
    ASK_ENGLISH        = State()
    WAITING_DECISION   = State()
    WAITING_SLOT       = State()
    FINISHED           = State()

async def api_call(
    method: str,
    endpoint: str,
    json_data: Optional[Dict] = None,
    files: Optional[Dict] = None,
    params: Optional[Dict] = None
) -> Dict:
    headers = {"X-Recruiter-ID": RECRUITER_ID}
    url = f"{API_BASE}{endpoint}"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            if method.upper() == "GET":
                r = await client.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                if files:
                    r = await client.post(url, headers=headers, files=files, data=json_data)
                else:
                    r = await client.post(url, headers=headers, json=json_data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            error_text = e.response.text if e.response else str(e)
            raise Exception(f"API error {e.response.status_code}: {error_text}") from e
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}") from e

def cancel_kb():
    kb = InlineKeyboardBuilder()
    kb.button(text="Отмена", callback_data="cancel")
    return kb.as_markup()

async def run_bot():
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def cmd_start(m: Message, state: FSMContext):
        try:
            jobs = await api_call("GET", "/api/jobs")
            if not jobs:
                await m.answer("Вакансий пока нет.")
                return

            kb = InlineKeyboardBuilder()
            for j in jobs:
                text = f"{j['title']} (порог {j['threshold_score']})"
                kb.button(text=text, callback_data=f"job:{j['id']}")
            kb.adjust(1)

            await m.answer("Выберите вакансию:", reply_markup=kb.as_markup())
            await state.set_state(CandidateFlow.SELECTING_JOB)
        except Exception as e:
            await m.answer(f"Не удалось загрузить вакансии\n{str(e)}")

    @dp.callback_query(F.data == "cancel")
    async def cancel_handler(c: CallbackQuery, state: FSMContext):
        await state.clear()
        await c.message.edit_text("Действие отменено. Начните заново командой /start")
        await c.answer()

    @dp.callback_query(F.data.startswith("job:"))
    async def job_selected(c: CallbackQuery, state: FSMContext):
        job_id = c.data.split(":", 1)[1]
        user = c.from_user

        payload = {
            "job_id": job_id,
            "telegram_user_id": user.id,
            "telegram_username": user.username or None
        }

        try:
            res = await api_call("POST", "/api/applications", json_data=payload)
            await state.update_data(
                application_id=res["application_id"],
                candidate_id=res.get("candidate_id")
            )
            await c.message.answer(
                "Отклик создан!\n\nПришлите резюме файлом (PDF или DOCX).",
                reply_markup=cancel_kb()
            )
            await state.set_state(CandidateFlow.WAITING_RESUME)
        except Exception as e:
            await c.message.answer(f"Ошибка создания отклика:\n{str(e)}")
        await c.answer()

    @dp.message(F.document, CandidateFlow.WAITING_RESUME)
    async def handle_resume(m: Message, state: FSMContext, bot: Bot):
        doc = m.document
        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        if doc.mime_type not in allowed_types:
            await m.answer("Поддерживаются только PDF и DOCX файлы.", reply_markup=cancel_kb())
            return

        data = await state.get_data()
        app_id = data.get("application_id")
        candidate_id = data.get("candidate_id")

        if not candidate_id:
            try:
                app_data = await api_call("GET", f"/api/applications/{app_id}")
                candidate_id = app_data.get("candidate_id")
                if not candidate_id:
                    raise ValueError("candidate_id не найден в отклике")
                await state.update_data(candidate_id=candidate_id)
            except Exception as e:
                await m.answer(f"Ошибка: не удалось определить кандидата.\n{str(e)}\nНачните заново /start")
                await state.clear()
                return

        try:
            file_info = await bot.get_file(doc.file_id)
            file_bytes = await bot.download_file(file_info.file_path)

            files = {
                "file": (doc.file_name or "resume.pdf", file_bytes.getvalue(), doc.mime_type)
            }

            upload_res = await api_call(
                "POST",
                f"/api/candidates/{candidate_id}/documents",
                files=files
            )

            document_id = upload_res["document_id"]
            await state.update_data(document_id=document_id)

            await m.answer("Файл успешно загружен. Запускаю анализ резюме...", reply_markup=cancel_kb())
            await state.set_state(CandidateFlow.PROCESSING_RESUME)

            # коммент тк парсинга пока нет
            # await api_call("POST", "/api/parsing/run", json_data={"document_id": document_id})
            await asyncio.sleep(4)

            await m.answer(
                "Резюме обработано.\n\n"
                "Укажите ваши зарплатные ожидания (в рублях на руки в месяц):",
                reply_markup=cancel_kb()
            )
            await state.set_state(CandidateFlow.ASK_SALARY)

        except Exception as e:
            await m.answer(f"Ошибка при обработке резюме:\n{str(e)}", reply_markup=cancel_kb())
            await state.clear()

    @dp.message(CandidateFlow.ASK_SALARY)
    async def process_salary(m: Message, state: FSMContext):
        await state.update_data(salary_expectation=m.text.strip())
        await m.answer("Какой формат работы вас интересует?\n(офис / гибрид / удалённо)", reply_markup=cancel_kb())
        await state.set_state(CandidateFlow.ASK_WORK_FORMAT)

    @dp.message(CandidateFlow.ASK_WORK_FORMAT)
    async def process_format(m: Message, state: FSMContext):
        await state.update_data(work_format=m.text.strip())
        await m.answer("По какой причине рассматриваете смену работы?", reply_markup=cancel_kb())
        await state.set_state(CandidateFlow.ASK_REASON)

    @dp.message(CandidateFlow.ASK_REASON)
    async def process_reason(m: Message, state: FSMContext):
        await state.update_data(reason=m.text.strip())
        await m.answer("Какой у вас уровень английского? (A1–C2 или None)", reply_markup=cancel_kb())
        await state.set_state(CandidateFlow.ASK_ENGLISH)

    @dp.message(CandidateFlow.ASK_ENGLISH)
    async def process_english(m: Message, state: FSMContext):
        await state.update_data(english_level=m.text.strip())
        data = await state.get_data()
        app_id = data.get("application_id")

        await m.answer("Спасибо за ответы! Проверяем вашу кандидатуру...")

        await asyncio.sleep(3)
        score = 82
        threshold = 70

        if score >= threshold:
            slots = [
                {"slot_id": "s1", "label": "20 марта 11:00–11:30"},
                {"slot_id": "s2", "label": "20 марта 14:00–14:30"},
                {"slot_id": "s3", "label": "21 марта 10:00–10:30"},
            ]

            kb = InlineKeyboardBuilder()
            for s in slots:
                kb.button(text=s["label"], callback_data=f"slot:{s['slot_id']}")
            kb.button(text="Отмена", callback_data="cancel")
            kb.adjust(1)

            await m.answer(
                f"Score: {score}/{threshold}. Вы прошли первичный отбор!\n\n"
                "Выберите удобное время для интервью:",
                reply_markup=kb.as_markup()
            )
            await state.set_state(CandidateFlow.WAITING_SLOT)
        else:
            feedback = (
                "Спасибо за отклик! К сожалению, на данный момент ваш опыт "
                "не полностью соответствует требованиям вакансии. "
                "Будем рады рассмотреть вас в будущем."
            )
            await m.answer(feedback)
            await state.set_state(CandidateFlow.FINISHED)

    @dp.callback_query(F.data.startswith("slot:"))
    async def select_slot(c: CallbackQuery, state: FSMContext):
        slot_id = c.data.split(":", 1)[1]
        # коммент тк пока нет календаря
        # await api_call("POST", f"/api/applications/{app_id}/book_slot", json_data={"slot_id": slot_id})
        await c.message.edit_text(
            f"Слот {slot_id} забронирован!\n\n"
            "Мы свяжемся с вами для подтверждения. Удачи!"
        )
        await state.set_state(CandidateFlow.FINISHED)
        await c.answer()

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(run_bot())
    