# law-qa-rag

Прототип RAG-пайплайна для вопросов по российским федеральным нормативным актам.

## Быстрый локальный pipeline

Ниже основной воспроизводимый сценарий: от исходных документов до генерации ответа.

```bash
cp .env.example .env
# Заполните в .env локальный POSTGRES_PASSWORD и GIGACHAT_CREDENTIALS.
docker compose up -d

.venv/bin/python -m pip install -r requirements.txt

# Для существующей БД примените web-миграцию один раз:
.venv/bin/python scripts/db/apply_migration.py migrations/001_answers_web_fields.sql

# Парсинг и чанкинг описаны в docs/1_parsing.md и docs/2_chunking.md.
# После них должны появиться data/chunks_json/acts.jsonl и data/chunks_json/chunks.jsonl.

.venv/bin/python scripts/pipeline/load_corpus.py data/chunks_json \
  --reset

.venv/bin/python scripts/pipeline/embed_chunks.py \
  --settings settings.yaml \
  --batch-size 8
```

После этого можно проверять retrieval:

```bash
.venv/bin/python scripts/debug/debug_sparse_search.py "водные объекты общего пользования" \
  --limit 5

.venv/bin/python scripts/debug/debug_dense_search.py "водные объекты общего пользования" \
  --limit 5 \
  --device cpu

.venv/bin/python scripts/debug/debug_hybrid_search.py "водные объекты общего пользования" \
  --limit 5 \
  --device cpu
```

Для генерации ответа через GigaChat заполните `GIGACHAT_CREDENTIALS` в `.env`:

```bash
.venv/bin/python scripts/debug/check_gigachat.py --list-models

.venv/bin/python scripts/debug/generate_answer.py "Что такое водные объекты общего пользования?" \
  --device cpu
```

Если GigaChat падает на SSL handshake или self-signed certificate в локальной сети, временно задайте в `.env`:

```bash
GIGACHAT_VERIFY_SSL_CERTS=false
GIGACHAT_TIMEOUT=120
GIGACHAT_MAX_RETRIES=3
```

Это локальная настройка для разработки. В production лучше использовать корректный CA bundle через `GIGACHAT_CA_BUNDLE_FILE`.

## Веб-прототип

Веб-прототип использует тот же generation pipeline, что и CLI, и сохраняет успешные запросы в `queries`, `answers` и `answer_citations`.

```bash
.venv/bin/python -m uvicorn law_qa_rag.web.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

После запуска откройте `http://127.0.0.1:8000/`.
Если нужно принудительно выбрать устройство для dense retrieval, задайте `RAG_DEVICE=cpu`, `cuda` или `mps`.

CLI и web автоматически читают локальный `.env`. Если явно задан `DATABASE_URL`, используется он; иначе URL собирается из `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER` и `POSTGRES_PASSWORD`.

## Данные

Основные директории pipeline:

- `data/raw_docs/*` — исходные `.doc`/`.html` документы;
- `data/parsed_json/parsed_*` — результат парсинга отдельных групп документов;
- `data/parsed_json/parsed_all` — объединенный parsed corpus;
- `data/chunks_json` — `acts.jsonl`, `chunks.jsonl`, `chunk_manifest.json`;
- PostgreSQL/pgvector — таблицы `acts`, `chunks`, embeddings и полнотекстовый индекс.

`data/` не хранится в git как обычный исходный код.

## Конфигурация

Главный конфиг — `settings.yaml`.

В нем задаются:

- embedding-модель и размерность;
- параметры чанкинга;
- retrieval-метод: `sparse`, `dense` или `weighted_hybrid`;
- параметры GigaChat generation: модель, token budget, температура и лимит ответа;
- параметры локальной БД.

Retrieval в generation pipeline выбирается из `settings.yaml`, а не хардкодится в CLI или web:

- `sparse` — PostgreSQL full-text search по `chunks.search_vector`;
- `dense` — pgvector-поиск по query embedding, только по chunks с тем же `embedding_model`;
- `weighted_hybrid` — объединение sparse и dense candidates через weighted RRF.

Основные параметры retrieval:

- `retrieval.top_k` — сколько chunks попадет дальше в prompt;
- `retrieval.candidate_limit` — сколько кандидатов взять из sparse/dense перед объединением;
- `retrieval.rrf_k` — сглаживающий параметр RRF;
- `retrieval.sparse_weight` и `retrieval.dense_weight` — веса sparse/dense в `weighted_hybrid`.

Основные параметры LLM:

- `llm.model: null` — использовать модель GigaChat SDK по умолчанию;
- `llm.context_token_budget` — лимит prompt+context перед вызовом LLM;
- `llm.max_output_tokens` — максимальная длина ответа модели;
- `llm.prompt_version` — версия prompt, сейчас `answer_v1`.

Локальный порт PostgreSQL: `5433`. Внутри контейнера PostgreSQL слушает стандартный `5432`.

## Структура проекта

- `scripts/pipeline/` — CLI-скрипты этапов подготовки корпуса;
- `scripts/debug/` — CLI-скрипты для ручной проверки retrieval, GigaChat и генерации;
- `src/law_qa_rag/` — reusable-ядро для config-driven retrieval и генерации ответа;
- `sql/init/` — расширения и схема PostgreSQL;
- `docs/` — описание отдельных этапов;
- `tests/` — unit-тесты.

## Тесты

```bash
.venv/bin/python -m unittest discover -s tests
```

## Документация

- `docs/1_parsing.md`;
- `docs/2_chunking.md`;
- `docs/3_load_corpus.md`;
- `docs/4_embed_chunks.md`;
- `docs/5_generation.md`;
- `docs/6_web_prototype.md`.
