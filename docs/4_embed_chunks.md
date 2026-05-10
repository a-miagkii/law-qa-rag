# Расчет embeddings: `scripts/pipeline/embed_chunks.py`

`embed_chunks.py` считает векторные представления для chunks, которые уже лежат в PostgreSQL, и записывает результат в поля `chunks.embedding` и `chunks.embedding_model`.

Этот этап запускается после `docs/3_load_corpus.md`.

## Подготовка подключения

```bash
cp .env.example .env
# Заполните POSTGRES_PASSWORD в .env.
```

Скрипт автоматически читает `.env`: если `DATABASE_URL` не задан, он собирается из `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER` и `POSTGRES_PASSWORD`.

Модель и размерность берутся из `settings.yaml`:

```yaml
embedding:
  embedding_model: BAAI/bge-m3
  embedding_dim: 1024
```

В командах ниже `--settings settings.yaml` передается явно, чтобы запуск был привязан к репозиторному конфигу.

## Пробный запуск (`--dry-run`)

Пробный запуск загружает модель, берет небольшой batch из БД, считает embeddings и проверяет размерность, но ничего не записывает:

```bash
.venv/bin/python scripts/pipeline/embed_chunks.py \
  --settings settings.yaml \
  --dry-run \
  --limit 8 \
  --device cpu
```

Такой запуск нужен, чтобы быстро проверить зависимости, доступность БД и совместимость `embedding_dim`.

## Полный запуск

CPU-вариант:

```bash
.venv/bin/python scripts/pipeline/embed_chunks.py \
  --settings settings.yaml \
  --batch-size 8 \
  --device cpu
```

Если доступна CUDA или MPS, можно выбрать устройство явно:

```bash
.venv/bin/python scripts/pipeline/embed_chunks.py \
  --settings settings.yaml \
  --batch-size 16 \
  --device cuda
```

`--device auto` сам выбирает `cuda`, затем `mps`, затем `cpu`.

## Что делает скрипт

Скрипт работает только с chunks без embeddings:

```sql
SELECT id, text
FROM chunks
WHERE embedding IS NULL
ORDER BY id
LIMIT ...
```

Для каждого batch:

- кодирует тексты через `SentenceTransformer`;
- нормализует embeddings (`normalize_embeddings=True`);
- проверяет, что размерность совпадает с `embedding.embedding_dim`;
- записывает vector в PostgreSQL;
- сохраняет имя модели в `embedding_model`;
- печатает время расчета batch, общее время batch, прогресс и примерный ETA.

Если запуск прервался, повторный запуск продолжит с chunks, у которых `embedding IS NULL`.

## Полезные параметры

- `--settings` — путь к YAML-конфигу;
- `--db-url` — явный PostgreSQL URL; если не передан, используется `DATABASE_URL` или `POSTGRES_*` из `.env`;
- `--model` — временно переопределить модель из `settings.yaml`;
- `--embedding-dim` — временно переопределить ожидаемую размерность;
- `--batch-size` — сколько chunks кодировать за один batch;
- `--limit` — ограничить число chunks для тестового запуска;
- `--device` — `auto`, `cpu`, `cuda` или `mps`;
- `--dry-run` — ничего не записывать в БД.

## Проверка после расчета

Сколько chunks еще без embeddings:

```bash
source .env

PGPASSWORD="$POSTGRES_PASSWORD" psql \
  -h localhost \
  -p "${POSTGRES_PORT:-5433}" \
  -U "$POSTGRES_USER" \
  -d "$POSTGRES_DB" \
  -c "SELECT count(*) FROM chunks WHERE embedding IS NULL;"
```

Распределение embeddings по моделям:

```bash
PGPASSWORD="$POSTGRES_PASSWORD" psql \
  -h localhost \
  -p "${POSTGRES_PORT:-5433}" \
  -U "$POSTGRES_USER" \
  -d "$POSTGRES_DB" \
  -c "SELECT embedding_model, count(*) FROM chunks WHERE embedding IS NOT NULL GROUP BY embedding_model ORDER BY count(*) DESC;"
```

Для готового dense retrieval все chunks, участвующие в поиске, должны иметь `embedding IS NOT NULL`, а `embedding_model` должен совпадать с моделью из `settings.yaml`.

## Типовые проблемы

`Нужен URL БД` — передайте `--db-url` или заполните `DATABASE_URL`/`POSTGRES_*` в `.env`.

Ошибка размерности embeddings означает, что `embedding.embedding_dim` не соответствует выбранной модели.

Первый запуск может быть долгим: модель скачивается локально, а полный расчет embeddings зависит от числа chunks, устройства и `batch-size`.
