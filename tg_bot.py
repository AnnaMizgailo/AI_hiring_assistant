import os
import asyncio
import logging
import logging
import sys
import secrets
from typing import Any, Dict, Optional, List
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, CallbackQuery, Document
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
import httpx
from dotenv import load_dotenv

# Настройка логирования с принудительным выводом
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot_debug.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Принудительно сбрасываем логи
sys.stdout.reconfigure(line_buffering=True)

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is empty. Проверьте .env файл")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
) -> Dict:
    url = f"{API_BASE}{endpoint}"
    headers = {}  # никаких заголовков аутентификации по умолчанию

    async with httpx.AsyncClient(timeout=180.0) as client:
        if method.upper() == "GET":
            r = await client.get(url, headers=headers, params=params)
        elif method.upper() == "POST":
            if files:
                # Для загрузки файлов нужно отправлять multipart, headers не нужны
                r = await client.post(url, files=files, data=json_data or {}, params=params)
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
        status = await api_call("GET", f"/api/documents/{document_id}")
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
    """Получает список всех вакансий через публичный эндпоинт."""
    try:
        jobs = await api_call("GET", "/api/public/jobs")
        if isinstance(jobs, list):
            return jobs
        logger.warning(f"Unexpected response from public jobs: {jobs}")
    except Exception as e:
        logger.exception("Failed to load public jobs")
    return []


def cancel_kb():
    kb = InlineKeyboardBuilder()
    
    for slot in slots:
        book_id = generate_short_id()
        slot_id = slot.get('slot_id')
        if not slot_id:
            logger.error(f"Slot missing slot_id: {slot}")
            continue
            
        callback_data_store[book_id] = ("book", slot_id, application_id)
        
        start = slot.get("start_dt", "").replace("T", " ").replace("Z", "")[:16]
        end = slot.get("end_dt", "").replace("T", " ").replace("Z", "")[11:16]
        kb.button(
            text=f"{start} - {end}", 
            callback_data=f"cb:{book_id}"
        )
    
    kb.adjust(1)
    return kb.as_markup()

# Обработчики
async def run_bot() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is empty")
    
    bot = Bot(BOT_TOKEN)
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    
    @dp.message(CommandStart())
    async def start_command(message: Message):
        welcome_text = (
            "👋 Добро пожаловать в HR Assistant Bot!\n\n"
            "Я помогу вам откликнуться на вакансии и пройти процесс отбора.\n\n"
            "Выберите действие:"
        )
        await message.answer(welcome_text, reply_markup=main_menu_keyboard())
    
    @dp.message(Command("help"))
    async def help_command(message: Message):
        help_text = (
            "ℹ️ Справка по использованию:\n\n"
            "1. Нажмите 'Список вакансий' для просмотра доступных позиций\n"
            "2. Выберите интересующую вакансию\n"
            "3. Отправьте ваше резюме в формате PDF или DOCX\n"
            "4. Дождитесь результатов оценки\n"
            "5. Ответьте на несколько вопросов\n"
            "6. Запишитесь на собеседование"
        )
        await message.answer(help_text, reply_markup=back_keyboard())
    
    @dp.callback_query(F.data == "help")
    async def help_callback(callback: CallbackQuery):
        help_text = (
            "ℹ️ Справка по использованию:\n\n"
            "1. Нажмите 'Список вакансий' для просмотра доступных позиций\n"
            "2. Выберите интересующую вакансию\n"
            "3. Отправьте ваше резюме в формате PDF или DOCX\n"
            "4. Дождитесь результатов оценки\n"
            "5. Ответьте на несколько вопросов\n"
            "6. Запишитесь на собеседование"
        )
        await callback.message.edit_text(help_text, reply_markup=back_keyboard())
        await callback.answer()
    
    @dp.callback_query(F.data == "list_jobs")
    async def show_jobs(callback: CallbackQuery):
        try:
            jobs = await api_list_jobs()
            if not jobs:
                await callback.message.edit_text(
                    "😕 На данный момент нет доступных вакансий.",
                    reply_markup=back_keyboard()
                )
                await callback.answer()
                return
            
            text = "📋 Доступные вакансии:\n\n"
            for i, job in enumerate(jobs, 1):
                text += f"{i}. {job['title']}\n"
                text += f"   Порог прохождения: {job['threshold_score']}\n\n"
            
            await callback.message.edit_text(
                text,
                reply_markup=jobs_keyboard(jobs)
            )
            await callback.answer()
            
        except Exception as e:
            logger.error(f"Error showing jobs: {e}", exc_info=True)
            sys.stdout.flush()
            await callback.message.edit_text(
                "❌ Ошибка при загрузке вакансий. Попробуйте позже.",
                reply_markup=back_keyboard()
            )
            await callback.answer()
    
    @dp.callback_query(F.data == "back_to_main")
    async def back_to_main(callback: CallbackQuery):
        await callback.message.edit_text(
            "Главное меню. Выберите действие:",
            reply_markup=main_menu_keyboard()
        )
        await callback.answer()
    
    @dp.callback_query(F.data.startswith("cb:"))
    async def handle_callback(callback: CallbackQuery, state: FSMContext):
        """Единый обработчик для всех callback с данными"""
        short_id = callback.data.split(":", 1)[1]
        data = callback_data_store.pop(short_id, None)
        
        if not data:
            await callback.answer("Данные устарели, начните заново", show_alert=True)
            return
        
        action = data[0]
        
        if action == "job":
            job_id = data[1]
            await process_job_selection(callback, state, job_id)
            
        elif action == "book":
            slot_id, application_id = data[1], data[2]
            await process_book_slot(callback, state, slot_id, application_id)
    
    async def process_job_selection(callback: CallbackQuery, state: FSMContext, job_id: str):
        """Обработка выбора вакансии"""
        await callback.message.edit_text("🔄 Создаю отклик...")
        await callback.answer()
        
        try:
            job_details = await api_get_job_details(job_id)
            job_title = job_details.get("title", "Неизвестно")
            
            result = await api_create_application(
                job_id, 
                callback.from_user.id, 
                callback.from_user.username
            )
            
            await state.update_data(
                application_id=result["application_id"],
                candidate_id=result["candidate_id"],
                job_id=job_id,
                job_title=job_title
            )
            
            text = (
                f"✅ Отклик на вакансию '{job_title}' создан!\n\n"
                "📄 Отправьте ваше резюме в формате PDF или DOCX.\n"
                "Я проанализирую его и покажу результаты оценки."
            )
            
            await callback.message.edit_text(text)
            await state.set_state(ApplicationStates.waiting_for_resume)
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error creating application: {error_message}", exc_info=True)
            sys.stdout.flush()
            
            if "Connection" in error_message:
                user_message = (
                    "❌ Не удалось подключиться к серверу\n\n"
                    "Убедитесь, что сервер запущен."
                )
            elif "404" in error_message:
                user_message = (
                    "❌ Вакансия не найдена\n\n"
                    "Попробуйте выбрать другую вакансию."
                )
            else:
                user_message = "❌ Ошибка при создании отклика. Попробуйте позже."
            
            await callback.message.edit_text(
                user_message,
                reply_markup=back_keyboard()
            )
    
    async def process_book_slot(callback: CallbackQuery, state: FSMContext, slot_id: str, application_id: str):
        """Запись на собеседование"""
        try:
            result = await api_book_slot(application_id, slot_id)
            
            text = (
                "✅ Вы успешно записаны на собеседование!\n\n"
                f"📅 Дата и время: {result.get('start_dt', '').replace('T', ' ')[:16]}\n\n"
                "Ссылка на встречу будет отправлена вам дополнительно.\n"
                "Пожалуйста, подготовьтесь к собеседованию!"
            )
            
            await callback.message.edit_text(text)
            await callback.answer()
            await state.clear()
            
        except Exception as e:
            logger.error(f"Error booking slot: {e}", exc_info=True)
            sys.stdout.flush()
            await callback.answer("❌ Ошибка при записи на собеседование", show_alert=True)
    
    @dp.message(ApplicationStates.waiting_for_resume, F.document)
    async def handle_resume(message: Message, state: FSMContext):
        """Обработка загруженного резюме"""
        document: Document = message.document
        
        logger.info(f"Received document: {document.file_name}")
        sys.stdout.flush()
        
        # Проверяем формат файла
        if not (document.file_name.endswith('.pdf') or 
                document.file_name.endswith('.docx') or 
                document.file_name.endswith('.doc')):
            await message.answer(
                "❌ Пожалуйста, отправьте файл в формате PDF или DOCX."
            )
            return
        
        # Получаем данные из состояния
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

            await api_call("POST", "/api/parsing/run", json_data={"document_id": document_id})

            await m.answer("Резюме загружено. Начинаю обработку, это может занять до нескольких минут...")

            try:
                await wait_for_document_parsing(document_id, timeout_seconds=600, poll_interval=3.0)
            except TimeoutError:
                await m.answer("Резюме ещё обрабатывается слишком долго. Попробуйте немного позже.")
                return
            
            elif parse_result.get("status") == "STARTED":
                # Ждем завершения парсинга
                max_attempts = 30
                for attempt in range(max_attempts):
                    await asyncio.sleep(10)
                    
                    doc_status = await api_get_document_status(document_id)
                    
                    if doc_status.get("parse_status") == "DONE":
                        break
                    elif doc_status.get("parse_status") == "ERROR":
                        await status_msg.edit_text("❌ Ошибка при анализе резюме")
                        await state.clear()
                        return
                    elif attempt == max_attempts - 1:
                        await status_msg.edit_text("⏱️ Превышено время ожидания. Попробуйте позже.")
                        await state.clear()
                        return
            
            # Запускаем скоринг
            await status_msg.edit_text("🔄 Оцениваю соответствие вакансии...")
            
            score_result = await api_run_scoring(application_id)
            
            # Получаем результаты
            score = score_result.get("score", 0)
            rationale = score_result.get("rationale", "")
            missing = score_result.get("missing_requirements", [])
            
            # Получаем пороговый балл
            try:
                job_details = await api_get_job_details(job_id)
                threshold = job_details.get("threshold_score", 0)
            except:
                threshold = 0
            
            # Формируем результат
            if score >= threshold:
                result_text = (
                    f"✅ **Вы прошли первичный отбор!**\n\n"
                    f"Вакансия: {job_title}\n"
                    f"Ваш балл: {score} (порог: {threshold})\n\n"
                )
                
                if missing:
                    result_text += "Рекомендации по развитию:\n"
                    for req in missing[:3]:
                        result_text += f"• {req}\n"
                
                await status_msg.edit_text(result_text)
                
                # Переходим к скрининг-опросу
                await state.update_data(application_id=application_id)
                await state.set_state(ApplicationStates.waiting_for_salary)
                
                await message.answer(
                    "📝 Несколько коротких вопросов перед записью на собеседование:\n\n"
                    "1/4: Какие у вас ожидания по зарплате?"
                )
            else:
                result_text = (
                    f"❌ К сожалению, вы не прошли первичный отбор.\n\n"
                    f"Вакансия: {job_title}\n"
                    f"Ваш балл: {score} (порог: {threshold})\n\n"
                )
                
                if missing:
                    result_text += "Чего не хватило:\n"
                    for req in missing[:5]:
                        result_text += f"• {req}\n"
                
                await status_msg.edit_text(result_text)
                await state.clear()
                
                # Предлагаем другие вакансии
                jobs = await api_list_jobs()
                await message.answer(
                    "Попробуйте откликнуться на другие вакансии:",
                    reply_markup=jobs_keyboard(jobs)
                )
            
        except Exception as e:
            logger.error(f"Error processing resume: {e}", exc_info=True)
            await status_msg.edit_text(
                "❌ Произошла ошибка при обработке резюме. Попробуйте позже."
            )
            await state.clear()

    @dp.message(ApplicationStates.waiting_for_salary)
    async def handle_salary(message: Message, state: FSMContext):
        """Обработка ответа о зарплате"""
        salary = message.text.strip()
        await state.update_data(salary_expectation=salary)
        await state.set_state(ApplicationStates.waiting_for_work_format)
        
        await message.answer(
            "2/4: Какой формат работы предпочитаете?\n"
            "(офис / гибрид / удаленно)"
        )

    @dp.message(ApplicationStates.waiting_for_work_format)
    async def handle_work_format(message: Message, state: FSMContext):
        """Обработка ответа о формате работы"""
        work_format = message.text.strip()
        await state.update_data(work_format=work_format)
        await state.set_state(ApplicationStates.waiting_for_reason)
        
        await message.answer(
            "3/4: Почему рассматриваете новые возможности?"
        )

    @dp.message(ApplicationStates.waiting_for_reason)
    async def handle_reason(message: Message, state: FSMContext):
        """Обработка причины поиска"""
        reason = message.text.strip()
        await state.update_data(reason=reason)
        await state.set_state(ApplicationStates.waiting_for_english)
        
        await message.answer(
            "4/4: Ваш уровень английского?\n"
            "(например: A1, A2, B1, B2, C1, C2 или описание)"
        )

    @dp.message(ApplicationStates.waiting_for_english)
    async def handle_english(message: Message, state: FSMContext):
        """Обработка уровня английского и отправка всех ответов"""
        english = message.text.strip()
        data = await state.get_data()
        
        # Собираем все ответы
        answers = {
            "salary_expectation": data.get("salary_expectation"),
            "work_format": data.get("work_format"),
            "reason": data.get("reason"),
            "english_level": english
        }
        
        # Отправляем на сервер
        application_id = data.get("application_id")
        job_id = data.get("job_id")
        
        status_msg = await message.answer("📝 Сохраняю ваши ответы...")
        
        try:
            await api_save_screening_answers(application_id, answers)
            
            await status_msg.edit_text("✅ Спасибо за ответы!")
            
            # Получаем и показываем слоты
            slots = await api_get_available_slots(job_id)
            
            if slots:
                # Создаем клавиатуру со слотами
                kb = InlineKeyboardBuilder()
                for slot in slots:
                    book_id = generate_short_id()
                    slot_id = slot.get('slot_id')
                    if slot_id:
                        callback_data_store[book_id] = ("book", slot_id, application_id)
                        
                        start = slot.get("start_dt", "").replace("T", " ").replace("Z", "")[:16]
                        end = slot.get("end_dt", "").replace("T", " ").replace("Z", "")[11:16]
                        kb.button(text=f"{start} - {end}", callback_data=f"cb:{book_id}")
                
                kb.adjust(1)
                
                await state.set_state(ApplicationStates.booking_slot)
                await message.answer(
                    "📅 Выберите удобное время для собеседования:",
                    reply_markup=kb.as_markup()
                )
            else:
                await message.answer(
                    "😕 Пока нет доступных слотов для собеседования. "
                    "Мы свяжемся с вами позже."
                )
                await state.clear()
                
        except Exception as e:
            logger.error(f"Error saving screening answers: {e}")
            await status_msg.edit_text(
                "❌ Ошибка при сохранении ответов. Попробуйте позже."
            )
            await state.clear()

    async def api_get_document_status(document_id: str) -> Dict[str, Any]:
        """Получение статуса документа"""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                url = f"{API_BASE}/api/documents/{document_id}"
                r = await client.get(url)
                
                if r.status_code == 200:
                    return r.json()
                else:
                    return {"parse_status": "ERROR"}
        except Exception as e:
            logger.error(f"Error getting document status: {e}")
            return {"parse_status": "ERROR"}
    
    @dp.message(ApplicationStates.waiting_for_resume)
    async def handle_wrong_input(message: Message):
        """Обработка не-документов в состоянии ожидания резюме"""
        await message.answer(
            "Пожалуйста, отправьте файл с резюме в формате PDF или DOCX."
        )
    
    @dp.message()
    async def handle_unknown(message: Message):
        """Обработка неизвестных команд"""
        await message.answer(
            "Я не понимаю эту команду. Используйте /start для начала работы."
        )
    
    @dp.message(Command("diagnose"))
    async def diagnose_command(message: Message):
        """Диагностика подключения к API"""
        health = await check_api_health()
        
        if health["status"] == "ok":
            await message.answer(f"✅ {health['message']}")
        elif health["status"] == "warning":
            await message.answer(f"⚠️ {health['message']}")
        else:
            await message.answer(f"❌ {health['message']}")
    
    logger.info("Bot started")
    sys.stdout.flush()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(run_bot())
    