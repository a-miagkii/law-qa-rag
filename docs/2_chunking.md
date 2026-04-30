# Чанкинг: `scripts/chunk_json.py`

`chunk_json.py` преобразует parsed JSON с юридическими актами в JSONL-файлы, готовые для загрузки в RAG-пайплайн.

Запуск:

```bash
.venv/bin/python scripts/chunk_json.py data/parsed_json data/chunks_json
```

Результат:

- `acts.jsonl`;
- `chunks.jsonl`;
- `chunk_manifest.json`.

## Конфигурация

Параметры чанкинга лежат в `settings.yaml`.

```yaml
embedding:
  embedding_model: BAAI/bge-m3
  embedding_dim: 1024

chunking:
  default_profile: base
  min_chunk_size_tokens: 100
  include_act_title: true
  include_path: true
  long_node_min_body_tokens: 50
  header_reserve_tokens: 80

  small:
    chunk_size_tokens: 500
    overlap_tokens: 80

  base:
    chunk_size_tokens: 800
    overlap_tokens: 120

  large:
    chunk_size_tokens: 1200
    overlap_tokens: 150
```

По умолчанию скрипт читает репозиторный `settings.yaml` и использует профиль из `chunking.default_profile`. Другой профиль можно выбрать явно:

```bash
.venv/bin/python scripts/chunk_json.py data/parsed_json data/chunks_json --chunk-profile large
```

CLI-флаги переопределяют значения из YAML:

```bash
.venv/bin/python scripts/chunk_json.py data/parsed_json data/chunks_json \
  --chunk-profile base \
  --chunk-size 900 \
  --chunk-overlap 120
```

Поддерживаемые флаги:

- `--settings`;
- `--chunk-profile`;
- `--embedding-model`;
- `--chunk-size`;
- `--chunk-overlap`;
- `--min-chunk-size`;
- `--no-act-title`;
- `--no-path`.

## Главное правило

Чанкер сохраняет точность юридических ссылок:

- одна статья становится одним chunk, если помещается в `chunk_size_tokens`;
- если статья слишком большая, она режется по целым paragraph nodes;
- короткие соседние статьи не склеиваются, потому что точная ссылка на статью важнее полного заполнения каждого chunk;
- overlap строится из последних целых абзацев предыдущего chunk и ограничивается `overlap_tokens`.

Если один абзац длиннее доступного body-бюджета, он режется по tokenizer offsets. Это fallback для необычно длинных абзацев.

## Текст chunk

Каждый chunk может включать три блока:

1. название акта;
2. структурный путь, например главу и заголовок статьи;
3. текст нормы.

Пример:

```text
Водный кодекс Российской Федерации

Глава 1. Общие положения
Статья 2. Водное законодательство

1. Водное законодательство состоит из настоящего Кодекса...
2. Нормы, регулирующие отношения...
```

Поля `include_act_title` и `include_path` управляют первыми двумя блоками.

## Выходные записи

`acts.jsonl` содержит очищенные метаданные актов для загрузки в базу.

`chunks.jsonl` содержит один JSON-объект на chunk:

- `canonical_key`;
- `chunk_index`;
- `text`;
- `structure_ref`;
- `article_no`;
- `clause_range`;
- `source_anchors`;
- `start_node_order`;
- `end_node_order`;
- `token_count`;
- `hash`;
- `oversized`;
- `undersized`.

`undersized` это диагностический флаг на основе `min_chunk_size_tokens`. Чанкер по-прежнему не склеивает короткие статьи.

## Валидация

Чанкер валидирует итоговый список chunks и записывает до 100 предупреждений в `chunk_manifest.json`.

Текущие проверки:

- нет `canonical_key`;
- нет `chunk_index`;
- пустой текст;
- некорректный `token_count`;
- нет `hash` или найден дублирующийся `hash`;
- некорректный диапазон node order;
- `chunk_index` внутри акта идет не подряд.

## Технические детали

Подсчет токенов использует tokenizer той же модели, которая указана для embeddings в `settings.yaml`. Tokenizer хранится в runtime-контексте, а результаты подсчета токенов кэшируются во время запуска.

Диапазоны пунктов сравниваются как dotted integer tuples, а не как float. Это предотвращает неправильный порядок для значений вроде `12.10`.
