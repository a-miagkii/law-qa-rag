# Веб-прототип: `law_qa_rag.web.app`

Веб-прототип — server-rendered интерфейс поверх текущего RAG pipeline. Он использует retrieval и LLM-настройки из `settings.yaml`, GigaChat provider из `src/law_qa_rag/llm` и сохраняет успешные ответы в PostgreSQL.

## Запуск

Сначала должен быть подготовлен корпус:

1. parsed JSON и chunks из `docs/1_parsing.md` и `docs/2_chunking.md`;
2. загруженные `acts` и `chunks` из `docs/3_load_corpus.md`;
3. рассчитанные embeddings из `docs/4_embed_chunks.md`, если retrieval использует `dense` или `weighted_hybrid`.

Локальные переменные лежат в `.env`:

```bash
cp .env.example .env
# Заполните POSTGRES_PASSWORD и GIGACHAT_CREDENTIALS.
```

Web-приложение автоматически читает `.env`. Если `DATABASE_URL` не задан, он собирается из `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER` и `POSTGRES_PASSWORD`.

Для существующей БД, созданной до web-этапа, один раз примените миграцию:

```bash
.venv/bin/python scripts/db/apply_migration.py migrations/001_answers_web_fields.sql
```

Эта миграция добавляет к таблице `answers` web/retrieval-поля:

- `needs_clarification`;
- `retrieval_method`;
- `retrieved_chunk_ids`;
- `dropped_chunk_ids`.

Запуск:

```bash
.venv/bin/python -m uvicorn law_qa_rag.web.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

Откройте `http://127.0.0.1:8000/`.

## Endpoint `/ask`

`POST /ask` принимает вопрос двумя способами.

HTML form:

```bash
curl -i \
  -X POST \
  -F "question=Что такое водные объекты общего пользования?" \
  http://127.0.0.1:8000/ask
```

Для формы успешный ответ — `303 See Other` на `/answers/{answer_id}`.

JSON:

```bash
curl \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"question":"Что такое водные объекты общего пользования?"}' \
  http://127.0.0.1:8000/ask
```

Для JSON endpoint возвращает `answer_id`, `answer`, `used_chunk_ids`, `answer_citations`, retrieval metadata и `latency_ms`.

## Страницы

- `/` — поле вопроса, кнопка отправки и примеры вопросов;
- `/answers/{answer_id}` — вопрос, ответ, предупреждение при нехватке контекста и список цитат справа;
- `/sources/{act_id}?answer_id=...` — реквизиты акта и chunks этого акта, процитированные chunks подсвечены.

На страницах выводится дисклеймер: ответ строится только по редакциям документов, загруженным в локальный корпус, и актуальность нужно проверять по официальному источнику.

## Сохранение в БД

Веб-прототип использует технического пользователя `external_uid=local-web`.

Перед сохранением приложение не меняет схему БД. Для существующей БД schema update выполняется явно через `migrations/001_answers_web_fields.sql`.

Успешный запуск сохраняет:

- `queries.question` и `queries.normalized_question`;
- `answers.answer_text`, `llm_model`, `prompt_version`, `latency_ms` и retrieval metadata;
- `answer_citations.chunk_id`, `rank`, `relevance_score`, `quote`.

Ошибки GigaChat или БД показываются пользователю, но в v1 отдельно не логируются.
