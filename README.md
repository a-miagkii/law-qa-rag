# law-qa-rag

Прототип RAG-пайплайна для вопросов по российским федеральным нормативным актам.

Локальный пайплайн:

```bash
.venv/bin/python scripts/parse_docs.py data/raw_codes data/parsed_json
.venv/bin/python scripts/chunk_json.py data/parsed_json data/chunks_json
```

Конфигурация находится в `settings.yaml`. Профили чанкинга лежат в `chunking.small`, `chunking.base` и `chunking.large`.

Тесты:

```bash
.venv/bin/python -m unittest discover -s tests
```

Подробнее:

- `docs/1_parsing.md`;
- `docs/2_chunking.md`.
