# law-qa-rag

Прототип RAG-пайплайна для вопросов по российским федеральным нормативным актам.

## Быстрый локальный pipeline

Ниже основной воспроизводимый сценарий: от исходных документов до генерации ответа.

```bash
cp .env.example .env
# Укажите локальный пароль в .env:
# POSTGRES_PASSWORD=<ваш_локальный_пароль>
source .env
export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT:-5433}/${POSTGRES_DB}"
docker compose up -d

.venv/bin/python -m pip install -r requirements.txt

# Парсинг и чанкинг описаны в docs/1_parsing.md и docs/2_chunking.md.
# После них должны появиться data/chunks_json/acts.jsonl и data/chunks_json/chunks.jsonl.

.venv/bin/python scripts/pipeline/load_corpus.py data/chunks_json \
  --db-url "$DATABASE_URL" \
  --reset

.venv/bin/python scripts/pipeline/embed_chunks.py \
  --settings settings.yaml \
  --db-url "$DATABASE_URL" \
  --batch-size 8
```

После этого можно проверять retrieval:

```bash
.venv/bin/python scripts/debug/debug_sparse_search.py "водные объекты общего пользования" \
  --db-url "$DATABASE_URL" \
  --limit 5

.venv/bin/python scripts/debug/debug_dense_search.py "водные объекты общего пользования" \
  --db-url "$DATABASE_URL" \
  --limit 5 \
  --device cpu

.venv/bin/python scripts/debug/debug_hybrid_search.py "водные объекты общего пользования" \
  --db-url "$DATABASE_URL" \
  --limit 5 \
  --device cpu
```

Для генерации ответа через GigaChat нужны credentials:

```bash
export GIGACHAT_CREDENTIALS="..."

.venv/bin/python scripts/debug/check_gigachat.py --list-models

.venv/bin/python scripts/debug/generate_answer.py "Что такое водные объекты общего пользования?" \
  --db-url "$DATABASE_URL" \
  --device cpu
```

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
- параметры GigaChat generation;
- параметры локальной БД.

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
- `docs/4_embed_chunks.md`.
