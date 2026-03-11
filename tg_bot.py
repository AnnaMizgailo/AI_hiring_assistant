import asyncio
import logging
import os
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Document, InlineKeyboardMarkup, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))
UPLOAD_TIMEOUT = float(os.getenv("UPLOAD_TIMEOUT", "120"))
BOOKING_TIMEOUT = float(os.getenv("BOOKING_TIMEOUT", "90"))

PARSING_TIMEOUT_SECONDS = int(os.getenv("PARSING_TIMEOUT_SECONDS", "600"))
PARSING_POLL_INTERVAL = float(os.getenv("PARSING_POLL_INTERVAL", "3"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CandidateFlow(StatesGroup):
    selecting_job = State()
    waiting_resume = State()
    ask_salary = State()
    ask_work_format = State()
    ask_reason = State()
    ask_english = State()
    choosing_day = State()
    choosing_slot = State()


def build_timeout(total_seconds: float | None) -> httpx.Timeout | None:
    if total_seconds is None:
        return None
    return httpx.Timeout(
        connect=min(20.0, total_seconds),
        read=total_seconds,
        write=total_seconds,
        pool=min(20.0, total_seconds),
    )


async def api_call(
    method: str,
    endpoint: str,
    *,
    json_data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout_seconds: Optional[float] = REQUEST_TIMEOUT,
) -> Any:
    url = f"{API_BASE}{endpoint}"
    timeout = build_timeout(timeout_seconds)

    async with httpx.AsyncClient(timeout=timeout) as client:
        if method.upper() == "GET":
            response = await client.get(url, params=params)
        elif method.upper() == "POST":
            if files is not None:
                response = await client.post(url, files=files, data=json_data or {}, params=params)
            else:
                response = await client.post(url, json=json_data, params=params)
        elif method.upper() == "PATCH":
            response = await client.patch(url, json=json_data, params=params)
        else:
            raise ValueError(f"Unsupported method: {method}")

    response.raise_for_status()
    if not response.content:
        return {}
    return response.json()


async def wait_for_document_parsing(document_id: str) -> Dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + PARSING_TIMEOUT_SECONDS
    last_status: Dict[str, Any] = {}

    while asyncio.get_running_loop().time() < deadline:
        status = await api_call("GET", f"/api/documents/{document_id}", timeout_seconds=REQUEST_TIMEOUT)
        last_status = status or {}
        parse_status = str(last_status.get("parse_status", "")).upper()

        if parse_status == "DONE":
            return last_status
        if parse_status == "ERROR":
            message = last_status.get("last_error") or "Не удалось обработать резюме."
            raise RuntimeError(message)

        await asyncio.sleep(PARSING_POLL_INTERVAL)

    raise TimeoutError(
        f"Парсинг резюме не завершился за {PARSING_TIMEOUT_SECONDS} секунд. Последний статус: {last_status}"
    )


async def load_jobs_for_candidate() -> List[Dict[str, Any]]:
    jobs = await api_call("GET", "/api/public/jobs", timeout_seconds=REQUEST_TIMEOUT)
    return jobs if isinstance(jobs, list) else []


def main_menu_keyboard() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="📋 Список вакансий", callback_data="list_jobs")
    kb.button(text="ℹ️ Помощь", callback_data="help")
    kb.adjust(1)
    return kb.as_markup()


def back_to_main_keyboard() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="◀️ Назад", callback_data="back_to_main")
    kb.adjust(1)
    return kb.as_markup()


def cancel_keyboard() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Отмена", callback_data="cancel")
    kb.adjust(1)
    return kb.as_markup()


def jobs_keyboard(jobs: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for job in jobs:
        kb.button(
            text=f"{job.get('title', 'Без названия')} (порог {job.get('threshold_score', '—')})",
            callback_data=f"job:{job['id']}",
        )
    kb.button(text="◀️ Назад", callback_data="back_to_main")
    kb.adjust(1)
    return kb.as_markup()


def build_days_keyboard(slots: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
    day_map: "OrderedDict[str, str]" = OrderedDict()
    for slot in slots:
        start_dt = datetime.fromisoformat(slot["start_dt"].replace("Z", "+00:00"))
        day_key = start_dt.strftime("%Y-%m-%d")
        day_label = start_dt.strftime("%d.%m.%Y")
        day_map.setdefault(day_key, day_label)

    kb = InlineKeyboardBuilder()
    for day_key, day_label in day_map.items():
        kb.button(text=day_label, callback_data=f"day:{day_key}")
    kb.button(text="Отмена", callback_data="cancel")
    kb.adjust(1)
    return kb.as_markup()


def build_slots_keyboard(slots: List[Dict[str, Any]], day_key: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for slot in slots:
        start_dt = datetime.fromisoformat(slot["start_dt"].replace("Z", "+00:00"))
        if start_dt.strftime("%Y-%m-%d") != day_key:
            continue
        end_dt = datetime.fromisoformat(slot["end_dt"].replace("Z", "+00:00"))
        label = f"{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}"
        kb.button(text=label, callback_data=f"slot:{slot['slot_id']}")
    kb.button(text="⬅️ Назад к дням", callback_data="choose_days")
    kb.button(text="Отмена", callback_data="cancel")
    kb.adjust(1)
    return kb.as_markup()


def format_rejection_message(decision_res: Dict[str, Any]) -> str:
    score = decision_res.get("score")
    threshold = decision_res.get("threshold_score")
    summary = decision_res.get("screening_summary") or "Краткое summary пока недоступно."
    feedback = decision_res.get("feedback") or "Спасибо за отклик. На текущем этапе мы не готовы продолжить процесс."

    parts = []
    if score is not None and threshold is not None:
        parts.append(f"Ваш результат: {score} из {threshold}.")
    parts.append(summary)
    parts.append(feedback)
    return "\n\n".join(part for part in parts if part)


async def run_bot() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is empty. Проверьте .env файл")

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())

    async def show_jobs_message(target: Message | CallbackQuery, state: FSMContext) -> None:
        jobs = await load_jobs_for_candidate()
        if not jobs:
            text = (
                "Сейчас нет доступных вакансий.\n\n"
                "Сначала рекрутер должен создать вакансию на сайте http://127.0.0.1:8000, "
                "после этого она появится здесь автоматически."
            )
            if isinstance(target, CallbackQuery):
                await target.message.edit_text(text, reply_markup=back_to_main_keyboard())
                await target.answer()
            else:
                await target.answer(text, reply_markup=main_menu_keyboard())
            await state.clear()
            return

        if isinstance(target, CallbackQuery):
            await target.message.edit_text(
                "Выберите вакансию для отклика:",
                reply_markup=jobs_keyboard(jobs),
            )
            await target.answer()
        else:
            await target.answer(
                "Выберите вакансию для отклика:",
                reply_markup=jobs_keyboard(jobs),
            )

        await state.update_data(available_jobs={job["id"]: job for job in jobs})
        await state.set_state(CandidateFlow.selecting_job)

    @dp.message(CommandStart())
    async def start_command(message: Message, state: FSMContext) -> None:
        await state.clear()
        await message.answer(
            "👋 Добро пожаловать в HR Assistant Bot!\n\n"
            "Здесь вы можете выбрать актуальную вакансию, загруженную рекрутером через веб-интерфейс, "
            "отправить резюме и пройти первичный отбор.",
            reply_markup=main_menu_keyboard(),
        )

    @dp.message(Command("help"))
    async def help_command(message: Message) -> None:
        await message.answer(
            "Как это работает:\n"
            "1. Рекрутер создаёт вакансию на сайте.\n"
            "2. Вы выбираете одну из доступных вакансий в боте.\n"
            "3. Отправляете резюме.\n"
            "4. Проходите короткий скрининг.\n"
            "5. Система принимает решение именно по выбранной вакансии.\n"
            "6. При успехе вы выбираете день и время собеседования.",
            reply_markup=back_to_main_keyboard(),
        )

    @dp.callback_query(F.data == "help")
    async def help_callback(callback: CallbackQuery) -> None:
        await callback.message.edit_text(
            "Как это работает:\n"
            "1. Рекрутер создаёт вакансию на сайте.\n"
            "2. Вы выбираете одну из доступных вакансий в боте.\n"
            "3. Отправляете резюме.\n"
            "4. Проходите короткий скрининг.\n"
            "5. Система принимает решение именно по выбранной вакансии.\n"
            "6. При успехе вы выбираете день и время собеседования.",
            reply_markup=back_to_main_keyboard(),
        )
        await callback.answer()

    @dp.callback_query(F.data == "back_to_main")
    async def back_to_main(callback: CallbackQuery, state: FSMContext) -> None:
        await state.clear()
        await callback.message.edit_text(
            "Главное меню. Выберите действие:",
            reply_markup=main_menu_keyboard(),
        )
        await callback.answer()

    @dp.callback_query(F.data == "cancel")
    async def cancel_handler(callback: CallbackQuery, state: FSMContext) -> None:
        await state.clear()
        await callback.message.edit_text(
            "Действие отменено. Можно начать заново через /start или открыть список вакансий.",
            reply_markup=main_menu_keyboard(),
        )
        await callback.answer()

    @dp.callback_query(F.data == "list_jobs")
    async def show_jobs_callback(callback: CallbackQuery, state: FSMContext) -> None:
        await show_jobs_message(callback, state)

    @dp.callback_query(F.data.startswith("job:"))
    async def job_selected(callback: CallbackQuery, state: FSMContext) -> None:
        job_id = callback.data.split(":", 1)[1]
        state_data = await state.get_data()
        available_jobs = state_data.get("available_jobs", {})
        selected_job = available_jobs.get(job_id)

        if not selected_job:
            jobs = await load_jobs_for_candidate()
            selected_job = next((job for job in jobs if job["id"] == job_id), None)

        if not selected_job:
            await callback.message.edit_text(
                "Вакансия не найдена. Возможно, она была удалена рекрутером. Выберите другую.",
                reply_markup=back_to_main_keyboard(),
            )
            await callback.answer()
            await state.clear()
            return

        payload = {
            "job_id": job_id,
            "telegram_user_id": callback.from_user.id,
            "telegram_username": callback.from_user.username or None,
        }

        try:
            result = await api_call(
                "POST",
                "/api/applications",
                json_data=payload,
                timeout_seconds=REQUEST_TIMEOUT,
            )
        except Exception as exc:
            logger.exception("Failed to create application")
            await callback.message.edit_text(
                f"Не удалось создать отклик: {exc}",
                reply_markup=back_to_main_keyboard(),
            )
            await callback.answer()
            await state.clear()
            return

        await state.update_data(
            application_id=result["application_id"],
            candidate_id=result["candidate_id"],
            job_id=job_id,
            job_title=selected_job.get("title", "Без названия"),
            threshold_score=selected_job.get("threshold_score"),
        )
        await state.set_state(CandidateFlow.waiting_resume)

        await callback.message.edit_text(
            f"✅ Отклик на вакансию «{selected_job.get('title', 'Без названия')}» создан.\n\n"
            "Теперь отправьте резюме одним файлом в формате PDF, DOC, DOCX или TXT.",
            reply_markup=cancel_keyboard(),
        )
        await callback.answer()

    @dp.message(CandidateFlow.waiting_resume, F.document)
    async def handle_resume(message: Message, state: FSMContext, bot: Bot) -> None:
        document: Document = message.document
        filename = document.file_name or "resume"
        allowed_extensions = (".pdf", ".doc", ".docx", ".txt")

        if not filename.lower().endswith(allowed_extensions):
            await message.answer(
                "Поддерживаются только PDF, DOC, DOCX или TXT файлы.",
                reply_markup=cancel_keyboard(),
            )
            return

        state_data = await state.get_data()
        candidate_id = state_data.get("candidate_id")
        if not candidate_id:
            await state.clear()
            await message.answer(
                "Не удалось найти данные отклика. Начните заново через /start.",
                reply_markup=main_menu_keyboard(),
            )
            return

        status_message = await message.answer("📥 Получил резюме. Загружаю и запускаю обработку...")

        try:
            file_info = await bot.get_file(document.file_id)
            file_buffer = await bot.download_file(file_info.file_path)
            file_bytes = file_buffer.read()

            upload_result = await api_call(
                "POST",
                f"/api/candidates/{candidate_id}/documents",
                files={"file": (filename, file_bytes, document.mime_type or "application/octet-stream")},
                timeout_seconds=UPLOAD_TIMEOUT,
            )
            document_id = upload_result["document_id"]
            await state.update_data(document_id=document_id)

            await api_call(
                "POST",
                "/api/parsing/run",
                json_data={"document_id": document_id},
                timeout_seconds=REQUEST_TIMEOUT,
            )

            await status_message.edit_text("🔄 Анализирую резюме. Это может занять некоторое время...")
            await wait_for_document_parsing(document_id)

            await status_message.edit_text("✅ Резюме обработано. Переходим к короткому скринингу.")
            await message.answer(
                "1/4. Какие у вас зарплатные ожидания?",
                reply_markup=cancel_keyboard(),
            )
            await state.set_state(CandidateFlow.ask_salary)
        except TimeoutError:
            await status_message.edit_text(
                "⏱️ Обработка резюме заняла слишком много времени. Попробуйте чуть позже."
            )
            await state.clear()
        except RuntimeError as exc:
            await status_message.edit_text(f"❌ Ошибка при обработке резюме: {exc}")
            await state.clear()
        except Exception as exc:
            logger.exception("Failed to handle resume")
            await status_message.edit_text(f"❌ Не удалось обработать резюме: {exc}")
            await state.clear()

    @dp.message(CandidateFlow.waiting_resume)
    async def handle_resume_wrong_type(message: Message) -> None:
        await message.answer(
            "Пожалуйста, отправьте резюме именно файлом.",
            reply_markup=cancel_keyboard(),
        )

    @dp.message(CandidateFlow.ask_salary)
    async def process_salary(message: Message, state: FSMContext) -> None:
        await state.update_data(salary_expectation=(message.text or "").strip())
        await message.answer(
            "1/4 сохранено.\n\n2/4. Какой формат работы вам подходит? (офис / гибрид / удалённо)",
            reply_markup=cancel_keyboard(),
        )
        await state.set_state(CandidateFlow.ask_work_format)

    @dp.message(CandidateFlow.ask_work_format)
    async def process_work_format(message: Message, state: FSMContext) -> None:
        await state.update_data(work_format=(message.text or "").strip())
        await message.answer(
            "2/4 сохранено.\n\n3/4. Почему вас заинтересовала эта вакансия?",
            reply_markup=cancel_keyboard(),
        )
        await state.set_state(CandidateFlow.ask_reason)

    @dp.message(CandidateFlow.ask_reason)
    async def process_reason(message: Message, state: FSMContext) -> None:
        await state.update_data(reason=(message.text or "").strip())
        await message.answer(
            "3/4 сохранено.\n\n4/4. Какой у вас уровень английского?",
            reply_markup=cancel_keyboard(),
        )
        await state.set_state(CandidateFlow.ask_english)

    @dp.message(CandidateFlow.ask_english)
    async def process_english(message: Message, state: FSMContext) -> None:
        english_level = (message.text or "").strip()
        await state.update_data(english_level=english_level)
        data = await state.get_data()
        application_id = data["application_id"]

        screening_answers = {
            "salary_expectation": data.get("salary_expectation"),
            "work_format": data.get("work_format"),
            "reason": data.get("reason"),
            "english_level": english_level,
        }

        status_message = await message.answer(
            "📝 Сохраняю ответы и запускаю финальную обработку заявки.\n"
            "Это может занять некоторое время — дождитесь ответа."
        )

        try:
            await api_call(
                "PATCH",
                f"/api/applications/{application_id}/screening-answers",
                json_data=screening_answers,
                timeout_seconds=REQUEST_TIMEOUT,
            )

            decision_res = await api_call(
                "POST",
                f"/api/applications/{application_id}/decision",
                timeout_seconds=None,
            )
        except Exception as exc:
            logger.exception("Failed after screening")
            await status_message.edit_text(f"❌ Не удалось обработать анкету: {exc}")
            await state.clear()
            return

        decision = decision_res.get("decision")
        score = decision_res.get("score")
        threshold = decision_res.get("threshold_score")
        summary = decision_res.get("screening_summary") or "Краткое summary пока недоступно."
        job_title = data.get("job_title", "выбранная вакансия")

        if decision == "INTERVIEW":
            slots = decision_res.get("slots") or []
            if not slots:
                await status_message.edit_text(
                    f"✅ По вакансии «{job_title}» вы прошли первичный отбор.\n\n"
                    f"Score: {score} из {threshold}.\n\n"
                    f"{summary}\n\n"
                    "Сейчас свободных слотов нет. Попробуйте позже."
                )
                await state.clear()
                return

            await state.update_data(available_slots=slots)
            await status_message.edit_text(
                f"✅ По вакансии «{job_title}» вы прошли первичный отбор.\n\n"
                f"Score: {score} из {threshold}.\n\n"
                f"{summary}\n\n"
                "Выберите день для собеседования:"
            )
            await message.answer(
                "Доступны только свободные интервалы по этой вакансии и календарю рекрутера.",
                reply_markup=build_days_keyboard(slots),
            )
            await state.set_state(CandidateFlow.choosing_day)
            return

        await status_message.edit_text(format_rejection_message(decision_res))
        await state.clear()

    @dp.callback_query(F.data == "choose_days")
    async def choose_days_again(callback: CallbackQuery, state: FSMContext) -> None:
        data = await state.get_data()
        slots = data.get("available_slots") or []
        if not slots:
            await callback.message.edit_text(
                "Список слотов устарел. Начните заново через /start.",
                reply_markup=main_menu_keyboard(),
            )
            await state.clear()
            await callback.answer()
            return

        await callback.message.edit_text(
            "Выберите день для собеседования:",
            reply_markup=build_days_keyboard(slots),
        )
        await state.set_state(CandidateFlow.choosing_day)
        await callback.answer()

    @dp.callback_query(CandidateFlow.choosing_day, F.data.startswith("day:"))
    async def choose_day(callback: CallbackQuery, state: FSMContext) -> None:
        day_key = callback.data.split(":", 1)[1]
        data = await state.get_data()
        slots = data.get("available_slots") or []
        if not slots:
            await callback.message.edit_text(
                "Список слотов устарел. Начните заново через /start.",
                reply_markup=main_menu_keyboard(),
            )
            await state.clear()
            await callback.answer()
            return

        await state.update_data(selected_day=day_key)
        await callback.message.edit_text(
            "Выберите удобное время:",
            reply_markup=build_slots_keyboard(slots, day_key),
        )
        await state.set_state(CandidateFlow.choosing_slot)
        await callback.answer()

    @dp.callback_query(CandidateFlow.choosing_slot, F.data.startswith("slot:"))
    async def choose_slot(callback: CallbackQuery, state: FSMContext) -> None:
        slot_id = callback.data.split(":", 1)[1]
        data = await state.get_data()
        application_id = data.get("application_id")
        slots = data.get("available_slots") or []
        selected_slot = next((slot for slot in slots if slot["slot_id"] == slot_id), None)

        if not application_id:
            await callback.message.edit_text(
                "Не удалось найти данные отклика. Начните заново через /start.",
                reply_markup=main_menu_keyboard(),
            )
            await state.clear()
            await callback.answer()
            return

        try:
            booking = await api_call(
                "POST",
                f"/api/applications/{application_id}/book-slot",
                json_data={"slot_id": slot_id},
                timeout_seconds=BOOKING_TIMEOUT,
            )
        except Exception as exc:
            logger.exception("Failed to book slot")
            await callback.message.edit_text(
                f"Не удалось забронировать время: {exc}",
                reply_markup=build_slots_keyboard(slots, data.get("selected_day", "")),
            )
            await callback.answer()
            return

        label = selected_slot.get("label") if selected_slot else f"{booking.get('start_dt')} - {booking.get('end_dt')}"
        await callback.message.edit_text(
            "✅ Собеседование подтверждено!\n\n"
            f"Вакансия: {data.get('job_title', 'выбранная вакансия')}\n"
            f"Слот: {label}\n"
            f"Статус: {booking.get('status')}\n\n"
            "Событие уже передано в календарь рекрутера."
        )
        await state.clear()
        await callback.answer()

    @dp.message()
    async def fallback_message(message: Message, state: FSMContext) -> None:
        current_state = await state.get_state()
        if current_state is None:
            await message.answer(
                "Используйте /start, чтобы увидеть актуальные вакансии, созданные рекрутером на сайте.",
                reply_markup=main_menu_keyboard(),
            )
        else:
            await message.answer(
                "Сообщение не распознано в текущем шаге. Следуйте сценарию или нажмите «Отмена».",
                reply_markup=cancel_keyboard(),
            )

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(run_bot())