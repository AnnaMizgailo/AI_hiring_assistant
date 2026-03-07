import os
import asyncio
from typing import Dict, Optional, Any, List

from dotenv import load_dotenv

load_dotenv()

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
import httpx

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
RECRUITER_ID = os.getenv("RECRUITER_ID", "demo-recruiter")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is empty. Проверьте .env файл")


class CandidateFlow(StatesGroup):
    SELECTING_JOB = State()
    WAITING_RESUME = State()
    ASK_SALARY = State()
    ASK_WORK_FORMAT = State()
    ASK_REASON = State()
    ASK_ENGLISH = State()
    WAITING_SLOT = State()
    FINISHED = State()


async def api_call(
    method: str,
    endpoint: str,
    json_data: Optional[Dict] = None,
    files: Optional[Dict] = None,
    params: Optional[Dict] = None,
    use_recruiter_header: bool = True,
) -> Dict:
    url = f"{API_BASE}{endpoint}"
    headers = {}
    if use_recruiter_header:
        headers["X-Recruiter-ID"] = RECRUITER_ID

    async with httpx.AsyncClient(timeout=180.0) as client:
        if method.upper() == "GET":
            r = await client.get(url, headers=headers, params=params)
        elif method.upper() == "POST":
            if files:
                r = await client.post(url, headers=headers, files=files, data=json_data or {}, params=params)
            else:
                r = await client.post(url, headers=headers, json=json_data, params=params)
        elif method.upper() == "PATCH":
            r = await client.patch(url, headers=headers, json=json_data, params=params)
        else:
            raise ValueError(f"Unsupported method: {method}")

        r.raise_for_status()
        return r.json()




async def wait_for_document_parsing(document_id: str, timeout_seconds: int = 600, poll_interval: float = 3.0) -> Dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_status = None

    while asyncio.get_running_loop().time() < deadline:
        status = await api_call("GET", f"/api/documents/{document_id}", use_recruiter_header=False)
        parse_status = (status.get("parse_status") or "").upper()
        last_status = status

        if parse_status == "DONE":
            return status
        if parse_status == "ERROR":
            message = status.get("last_error") or "Не удалось обработать резюме."
            raise RuntimeError(message)

        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Парсинг резюме не завершился за {timeout_seconds} секунд. Последний статус: {last_status}")


async def load_jobs_for_candidate() -> List[Dict[str, Any]]:
    """
    Сначала пытаемся получить публичный список вакансий для кандидата.
    Если такого эндпойнта ещё нет, используем текущий recruiter-scoped /api/jobs.
    """
    try:
        jobs = await api_call("GET", "/api/public/jobs", use_recruiter_header=False)
        if isinstance(jobs, list):
            return jobs
    except Exception:
        pass

    jobs = await api_call("GET", "/api/jobs", use_recruiter_header=True)
    return jobs if isinstance(jobs, list) else []


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
            jobs = await load_jobs_for_candidate()
            if not jobs:
                await m.answer(
                    "Вакансий пока нет.\n"
                    "Проверьте, что вакансия создана в веб-интерфейсе и сервер запущен."
                )
                return

            kb = InlineKeyboardBuilder()
            for j in jobs:
                threshold = j.get("threshold_score", "—")
                title = j.get("title", "Без названия")
                kb.button(text=f"{title} (порог {threshold})", callback_data=f"job:{j['id']}")
            kb.adjust(1)

            await m.answer("Выберите вакансию:", reply_markup=kb.as_markup())
            await state.set_state(CandidateFlow.SELECTING_JOB)
        except Exception as e:
            await m.answer(f"Не удалось загрузить вакансии.\n{str(e)}")

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
            "telegram_username": user.username or None,
        }
        try:
            res = await api_call("POST", "/api/applications", json_data=payload)
            await state.update_data(
                application_id=res["application_id"],
                candidate_id=res["candidate_id"],
                job_id=job_id,
            )
            await c.message.answer(
                "Отклик создан. Пришлите резюме файлом (PDF или DOCX).",
                reply_markup=cancel_kb(),
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
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ]
        if doc.mime_type not in allowed_types:
            await m.answer("Поддерживаются только PDF и DOCX файлы.", reply_markup=cancel_kb())
            return

        data = await state.get_data()
        candidate_id = data["candidate_id"]

        try:
            file_info = await bot.get_file(doc.file_id)
            file_bytes = await bot.download_file(file_info.file_path)

            files = {
                "file": (
                    doc.file_name or "resume.pdf",
                    file_bytes.getvalue(),
                    doc.mime_type,
                )
            }

            upload_res = await api_call("POST", f"/api/candidates/{candidate_id}/documents", files=files)
            document_id = upload_res["document_id"]
            await state.update_data(document_id=document_id)

            await api_call("POST", "/api/parsing/run", json_data={"document_id": document_id}, use_recruiter_header=False)

            await m.answer("Резюме загружено. Начинаю обработку, это может занять до нескольких минут...")

            try:
                await wait_for_document_parsing(document_id, timeout_seconds=600, poll_interval=3.0)
            except TimeoutError:
                await m.answer("Резюме ещё обрабатывается слишком долго. Попробуйте немного позже.")
                return
            except RuntimeError as e:
                await m.answer(f"Ошибка при обработке резюме: {e}")
                return
            except Exception as e:
                await m.answer(f"Ошибка при обработке резюме: {e}")
                return

            await m.answer("Резюме обработано. Продолжаем.")

            await m.answer(
                "Укажите ваши зарплатные ожидания (в рублях на руки в месяц):",
                reply_markup=cancel_kb(),
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
        english_level = m.text.strip()
        await state.update_data(english_level=english_level)

        data = await state.get_data()
        app_id = data["application_id"]

        await m.answer("Спасибо! Проверяем вашу кандидатуру...")

        try:
            await api_call(
                "PATCH",
                f"/api/applications/{app_id}/screening-answers",
                json_data={
                    "salary_expectation": data.get("salary_expectation"),
                    "work_format": data.get("work_format"),
                    "reason": data.get("reason"),
                    "english_level": english_level,
                },
            )

            decision_res = await api_call("POST", f"/api/applications/{app_id}/decision")

            score = decision_res.get("score")
            threshold = decision_res.get("threshold_score")
            summary = decision_res.get("screening_summary") or "Краткая сводка пока недоступна."

            if decision_res.get("decision") == "INTERVIEW":
                slots = decision_res.get("slots") or []
                if not slots:
                    await m.answer(
                        "Вы прошли отбор, но сейчас нет свободных слотов.\n"
                        "Мы свяжемся с вами позже."
                    )
                    await state.set_state(CandidateFlow.FINISHED)
                    return

                kb = InlineKeyboardBuilder()
                for s in slots:
                    kb.button(text=s["label"], callback_data=f"slot:{s['slot_id']}")
                kb.button(text="Отмена", callback_data="cancel")
                kb.adjust(1)

                await m.answer(
                    f"Score: {score}/{threshold}. Вы прошли первичный отбор!\n\n"
                    f"Краткая сводка: {summary}\n\n"
                    "Выберите удобное время для интервью:",
                    reply_markup=kb.as_markup(),
                )
                await state.set_state(CandidateFlow.WAITING_SLOT)

            else:
                feedback = decision_res.get("feedback") or (
                    "Спасибо за отклик! На этом этапе мы не можем продолжить процесс."
                )
                await m.answer(feedback)
                await state.set_state(CandidateFlow.FINISHED)

        except Exception as e:
            await m.answer(f"Ошибка при финальной обработке кандидатуры:\n{str(e)}")
            await state.clear()

    @dp.callback_query(F.data.startswith("slot:"))
    async def select_slot(c: CallbackQuery, state: FSMContext):
        slot_id = c.data.split(":", 1)[1]
        data = await state.get_data()
        app_id = data["application_id"]

        try:
            booking = await api_call(
                "POST",
                f"/api/applications/{app_id}/book-slot",
                json_data={"slot_id": slot_id},
            )
            await c.message.edit_text(
                "Интервью забронировано!\n\n"
                f"Дата начала: {booking['start_dt']}\n"
                f"Дата окончания: {booking['end_dt']}\n"
                f"ID события: {booking['event_id']}\n\n"
                "Если у рекрутера подключён реальный Google Calendar, встреча уже создана там."
            )
            await state.set_state(CandidateFlow.FINISHED)

        except Exception as e:
            await c.message.answer(f"Не удалось забронировать слот:\n{str(e)}")
            await state.clear()

        await c.answer()

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(run_bot())