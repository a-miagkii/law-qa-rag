# Retrieval и генерация ответа: `law_qa_rag.generate_answer`

Этот этап превращает найденные chunks в финальный ответ LLM с проверяемыми цитатами.

Общая схема:

```text
retrieval selected by settings.yaml
  -> token budget
  -> prompt answer_v1
  -> GigaChat JSON
  -> validation
  -> answer_citations
```

## Настройки retrieval

Retrieval выбирается из `settings.yaml`, поэтому один и тот же generation pipeline можно запускать с разными методами поиска:

```yaml
retrieval:
  method: weighted_hybrid
  top_k: 10
  candidate_limit: 50
  rrf_k: 60
  sparse_weight: 0.4
  dense_weight: 0.6
```

Поддерживаемые значения `retrieval.method`:

- `sparse` — PostgreSQL full-text search по `chunks.search_vector`;
- `dense` — pgvector-поиск по query embedding;
- `weighted_hybrid` — объединение sparse и dense candidates через weighted RRF.

Для `dense` и `weighted_hybrid` embeddings должны быть заранее рассчитаны командой из `docs/4_embed_chunks.md`.
Dense-поиск фильтрует chunks по `embedding_model`, чтобы query embedding сравнивался только с vectors той же модели.

## Weighted Hybrid

`weighted_hybrid` сначала получает candidates из sparse и dense поиска, затем объединяет их по формуле:

```text
score = sparse_weight / (rrf_k + sparse_rank)
      + dense_weight / (rrf_k + dense_rank)
```

Если chunk найден только одним методом, в score участвует только соответствующая часть формулы.
После fusion берутся первые `retrieval.top_k` chunks.

Практический смысл параметров:

- `candidate_limit` — ширина поиска перед объединением;
- `top_k` — сколько chunks попадет в prompt;
- `rrf_k` — насколько сильно сглаживать влияние ранга;
- `sparse_weight` и `dense_weight` — баланс лексического и семантического поиска.

## Настройки LLM

Параметры генерации тоже лежат в `settings.yaml`:

```yaml
llm:
  provider: gigachat
  model: null
  temperature: 0
  max_output_tokens: 1200
  context_token_budget: 6000
  prompt_version: answer_v1
```

Для локального запуска заполните в `.env`:

```bash
GIGACHAT_CREDENTIALS=...
```

`llm.model: null` означает, что используется модель GigaChat SDK по умолчанию.
Если указать конкретное имя модели, `check_gigachat.py` и `generate_answer.py` проверят, что модель доступна аккаунту.

`context_token_budget` ограничивает prompt вместе с найденным контекстом. Если все chunks не помещаются, низкоранговый хвост отбрасывается и его ids попадают в `dropped_chunk_ids`.

## Prompt `answer_v1`

Prompt v1 требует от модели:

- отвечать только по переданному контексту;
- не использовать внешние знания;
- при нехватке контекста честно сказать об этом;
- вернуть только JSON без markdown.

Ожидаемый JSON:

```json
{
  "answer": "ответ на русском языке",
  "used_chunk_ids": [123, 456],
  "needs_clarification": false
}
```

`used_chunk_ids` валидируются: если модель сослалась на chunk, которого не было в выбранном контексте, generation падает с ошибкой.

## Цитаты

`answer_citations` собираются детерминированно после ответа LLM.
Для каждого `used_chunk_id` берется полный текст соответствующего chunk:

- `chunk_id`;
- `rank`;
- `relevance_score`;
- `quote`;
- название акта, номер, дата;
- `structure_ref`, `article_no`, `clause_range`.

В v1 `quote` равен полному тексту chunk. Точное выделение span внутри chunk можно добавить отдельным этапом позже.

## CLI-проверки

Проверить связь и доступные модели:

```bash
.venv/bin/python scripts/debug/check_gigachat.py --list-models
```

Посчитать prompt/context tokens без вызова генерации:

```bash
.venv/bin/python scripts/debug/count_answer_tokens.py \
  "Что такое водные объекты общего пользования?" \
  --device cpu
```

Сгенерировать ответ:

```bash
.venv/bin/python scripts/debug/generate_answer.py \
  "Что такое водные объекты общего пользования?" \
  --device cpu
```

CLI автоматически читает `.env`. Если явно задан `DATABASE_URL`, используется он; иначе URL собирается из `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER` и `POSTGRES_PASSWORD`.

