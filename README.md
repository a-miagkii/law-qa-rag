# law-qa-rag

Локальный MVP RAG-системы вопрос-ответ по корпусу федеральных нормативных правовых актов РФ.

Проект сделан как выпускная квалификационная работа и как прикладной pet-project: в нем есть полный контур от подготовки правового корпуса до web-прототипа, сохранения ответов и экспериментальной оценки retrieval/generation.

## Что реализовано

- Парсинг `.doc`/`.html` нормативных актов в структурированный JSON.
- Чанкинг актов с сохранением юридической структуры: акт, глава, статья, пункты.
- Загрузка корпуса в PostgreSQL с `pgvector` и полнотекстовым индексом.
- Расчет embeddings через `BAAI/bge-m3`.
- Retrieval в трех режимах: `sparse`, `dense`, `weighted_hybrid`.
- Генерация ответа через GigaChat с жестким JSON-prompt: ответ только по найденному контексту.
- Детерминированные цитаты: `used_chunk_ids` превращаются в `answer_citations`.
- Server-rendered web-прототип на FastAPI + Jinja2.
- Локальная регистрация/вход, история вопросов пользователя и feedback по ответам.
- Экспериментальная оценка retrieval и generation в файловом контуре `eval/results`.

## Архитектура

```text
raw docs
  -> parse_docs
  -> parsed JSON
  -> chunk_json
  -> acts.jsonl / chunks.jsonl
  -> PostgreSQL + pgvector
  -> retrieval selected by settings.yaml
  -> token budget
  -> prompt answer_v1
  -> GigaChat JSON
  -> validation
  -> answer + citations
```

Основная логика находится в `src/law_qa_rag/`:

- `retrieval.py` - sparse/dense/weighted hybrid retrieval;
- `generation.py` - token budget, prompt call, JSON validation, citations;
- `llm/` - интерфейс LLM provider и GigaChat provider;
- `persistence.py` - сохранение пользователей, вопросов, ответов, цитат и feedback;
- `web/` - FastAPI/Jinja web-прототип.

CLI-скрипты разделены по назначению:

- `scripts/pipeline/` - подготовка корпуса;
- `scripts/debug/` - ручная проверка retrieval, GigaChat и generation;
- `scripts/eval/` - экспериментальная оценка.

## Результаты экспериментов

Retrieval оценивался на 50 вопросах. Сводные результаты лежат в `eval/results/retrieval/summary_metrics.csv`.

| Метод | Hit@10 | Recall@10 | Median latency |
| --- | ---: | ---: | ---: |
| sparse | 0.18 | 0.153 | 24 ms |
| dense | 0.88 | 0.844 | 216.5 ms |
| weighted_hybrid 0.4/0.6 | 0.88 | 0.830 | 145 ms |
| weighted_hybrid 0.5/0.5 | 0.92 | 0.865 | 136 ms |
| weighted_hybrid 0.3/0.7 | 0.88 | 0.834 | 132 ms |

Generation-прогон на 50 вопросах сохранен в:

- `eval/results/generation/generation_all_v1.jsonl`;
- `eval/results/generation/generation_all_v1.csv`.

Результаты экспериментов не хранятся в PostgreSQL. PostgreSQL используется для корпуса, фрагментов, пользователей, запросов, ответов, цитат и feedback. Экспериментальные артефакты хранятся как JSONL/CSV/YAML.

## Быстрый запуск

Требования:

- Python 3.11+;
- Docker;
- GigaChat credentials для генерации ответов;
- локально подготовленные документы в `data/raw_docs/*`.

```bash
cp .env.example .env
# Заполните POSTGRES_PASSWORD, SESSION_SECRET и GIGACHAT_CREDENTIALS.
docker compose up -d
```

Установка зависимостей:

```bash
uv sync --locked --dev
```

Если `uv` не используется:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Для существующей БД примените миграции:

```bash
.venv/bin/python scripts/db/apply_migration.py sql/migrations/001_answers_web_fields.sql
.venv/bin/python scripts/db/apply_migration.py sql/migrations/002_auth_feedback.sql
```

Подготовка корпуса описана подробнее в `docs/1_parsing.md` - `docs/4_embed_chunks.md`. Минимальный порядок:

```bash
.venv/bin/python scripts/pipeline/parse_docs.py data/raw_docs/codex data/parsed_json/parsed_codex
.venv/bin/python scripts/pipeline/chunk_json.py data/parsed_json/parsed_all data/chunks_json --settings settings.yaml
.venv/bin/python scripts/pipeline/load_corpus.py data/chunks_json --reset
.venv/bin/python scripts/pipeline/embed_chunks.py --settings settings.yaml --batch-size 8 --device cpu
```

Проверка GigaChat и генерации:

```bash
.venv/bin/python scripts/debug/check_gigachat.py --list-models
.venv/bin/python scripts/debug/generate_answer.py "Что такое водные объекты общего пользования?" --device cpu
```

Запуск web-прототипа:

```bash
.venv/bin/python -m uvicorn law_qa_rag.web.app:app --host 127.0.0.1 --port 8000 --reload
```

После запуска откройте `http://127.0.0.1:8000/`.

## Конфигурация

Главный файл настроек - `settings.yaml`.

Ключевые параметры:

- `embedding.embedding_model` - embedding-модель, сейчас `BAAI/bge-m3`;
- `retrieval.method` - `sparse`, `dense` или `weighted_hybrid`;
- `retrieval.top_k` - число фрагментов для prompt;
- `retrieval.candidate_limit` - число кандидатов до финального отбора;
- `llm.context_token_budget` - лимит контекста;
- `llm.max_output_tokens` - лимит ответа;
- `llm.prompt_version` - версия prompt, сейчас `answer_v1`.

CLI и web автоматически читают `.env`. Если задан `DATABASE_URL`, используется он; иначе URL собирается из `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`.

Если LLM возвращает невалидный JSON, сырой ответ сохраняется в `RAG_LLM_DEBUG_DIR` (`logs/llm_debug` по умолчанию). Каталог `logs/` игнорируется git.

Если GigaChat падает на SSL handshake или self-signed certificate в локальной сети, для локальной разработки можно задать:

```bash
GIGACHAT_VERIFY_SSL_CERTS=false
GIGACHAT_TIMEOUT=120
GIGACHAT_MAX_RETRIES=3
```

## Web-прототип

Web-интерфейс поддерживает:

- регистрацию и вход локального пользователя;
- отправку вопроса только после входа;
- страницу ответа с цитатами из первоисточников;
- страницу источника с подсветкой использованных фрагментов;
- профиль пользователя с историей вопросов;
- оценку ответа и комментарий.

Авторизация минимальная локальная. В проекте нет ролей, OAuth, восстановления пароля и промышленного разграничения доступа.

## Тесты

```bash
.venv/bin/python -m unittest discover -s tests
```

## Контур качества

В проекте используется `ruff` для форматирования и базового lint. Конфигурация находится в `pyproject.toml`, lock-файл зависимостей - `uv.lock`.

Форматирование:

```bash
uv run ruff format src scripts tests
```

Проверки перед коммитом:

```bash
uv run ruff format --check src scripts tests
uv run ruff check src scripts tests
uv run python -m unittest discover -s tests
```

GitHub Actions workflow `.github/workflows/ci.yml` запускает тот же набор проверок на `push` и `pull_request`: установка зависимостей через `uv sync --locked --dev`, проверка форматирования, lint и unit tests.

## Документация

- `docs/1_parsing.md` - парсинг документов;
- `docs/2_chunking.md` - чанкинг JSON;
- `docs/3_load_corpus.md` - загрузка корпуса в PostgreSQL;
- `docs/4_embed_chunks.md` - расчет embeddings;
- `docs/5_generation.md` - генерация ответа;
- `docs/6_web_prototype.md` - web-прототип;
- `docs/7_evaluation.md` - экспериментальная оценка.

## Ограничения

- Это локальный MVP, не production-сервис.
- Ответы строятся только по редакциям документов, загруженным в локальный корпус.
- Актуальность нормы нужно проверять по официальному источнику.
- GigaChat используется как внешний LLM provider, поэтому live generation требует credentials и сетевого доступа.
