# Загрузка корпуса: `scripts/pipeline/load_corpus.py`

`load_corpus.py` загружает подготовленные `acts.jsonl` и `chunks.jsonl` в PostgreSQL.
Это отдельный этап после чанкинга и до расчета embeddings.

## Вход и результат

Входная директория:

```text
data/chunks_json/
  acts.jsonl
  chunks.jsonl
  chunk_manifest.json
```

После успешной загрузки в БД заполнены таблицы:

- `acts` — метаданные нормативных актов;
- `chunks` — тексты chunks, структурные ссылки и служебные поля.

На этом этапе `chunks.embedding` и `chunks.embedding_model` остаются пустыми. Их заполняет следующий этап из `docs/4_embed_chunks.md`.

## Подготовка подключения

Локальный пароль задается в `.env`. Значение `change_me` из `.env.example` нужно заменить на свой локальный пароль.

```bash
source .env
export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT:-5433}/${POSTGRES_DB}"
docker compose up -d
```

Порт снаружи берется из `POSTGRES_PORT`; в текущей конфигурации это `5433`.

## Пробный запуск (`--dry-run`)

Перед записью в БД полезно проверить JSONL-файлы:

```bash
.venv/bin/python scripts/pipeline/load_corpus.py data/chunks_json --dry-run
```

Пробный запуск читает `acts.jsonl` и `chunks.jsonl`, проверяет обязательные поля, дубли и ссылочную связность между актами и chunks, но ничего не пишет в PostgreSQL.

## Полная загрузка

Обычный запуск для локальной БД:

```bash
.venv/bin/python scripts/pipeline/load_corpus.py data/chunks_json \
  --db-url "$DATABASE_URL"
```

Если нужно полностью пересобрать корпус в БД:

```bash
.venv/bin/python scripts/pipeline/load_corpus.py data/chunks_json \
  --db-url "$DATABASE_URL" \
  --reset
```

`--reset` очищает таблицы корпуса перед загрузкой. Без `--reset` скрипт работает в режиме замены для загружаемых актов: обновляет `acts`, удаляет старые chunks этих актов и вставляет актуальные chunks заново. Это важно, если после нового чанкинга у акта стало меньше chunks.

## Что проверяет скрипт

Для `acts.jsonl`:

- обязательные поля акта: `canonical_key`, `act_kind`, `doc_type`, `title`, `doc_number`, `doc_date`, `edition_as_of`, `source_file`;
- отсутствие дублей по `canonical_key`.

Для `chunks.jsonl`:

- каждый `canonical_key` должен быть в `acts.jsonl`;
- `chunk_index` должен быть целым числом `>= 0`;
- `text`, `token_count` и `hash` должны быть заполнены;
- пары `(canonical_key, chunk_index)` не должны повторяться;
- `hash` не должен повторяться внутри загружаемого корпуса;
- `source_anchors` должен быть списком;
- `end_node_order` не должен быть меньше `start_node_order`.

## Что пишет в БД

В `acts` загружаются нормализованные метаданные акта.

В `chunks` загружаются:

- `act_id`;
- `chunk_index`;
- `text`;
- `structure_ref`;
- `article_no`;
- `clause_range`;
- `source_anchors`;
- `start_node_order`;
- `end_node_order`;
- `token_count`;
- `hash`.

При обновлении chunk скрипт сбрасывает `embedding` и `embedding_model` в `NULL`, потому что текст изменился и старый embedding больше нельзя использовать.

## Проверка после загрузки

Быстрые SQL-проверки:

```bash
PGPASSWORD="$POSTGRES_PASSWORD" psql \
  -h localhost \
  -p "${POSTGRES_PORT:-5433}" \
  -U "$POSTGRES_USER" \
  -d "$POSTGRES_DB" \
  -c "SELECT count(*) AS acts FROM acts;"

PGPASSWORD="$POSTGRES_PASSWORD" psql \
  -h localhost \
  -p "${POSTGRES_PORT:-5433}" \
  -U "$POSTGRES_USER" \
  -d "$POSTGRES_DB" \
  -c "SELECT count(*) AS chunks, count(*) FILTER (WHERE embedding IS NULL) AS without_embeddings FROM chunks;"
```

После `load_corpus.py` нормальное состояние: `chunks` заполнены, а `without_embeddings` равен общему числу chunks.

## Типовые проблемы

`Missing file: data/chunks_json/acts.jsonl` — сначала выполните чанкинг из `docs/2_chunking.md`.

`Database URL is required` — передайте `--db-url` или задайте `DATABASE_URL`.

Ошибка дубля `hash` обычно означает, что в `chunks.jsonl` попали одинаковые тексты chunk. Нужно проверять результат чанкинга, а не чинить это на этапе загрузки.
