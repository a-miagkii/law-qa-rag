# Экспериментальная оценка

Документ описывает экспериментальную оценку RAG-системы: проверку качества retrieval и проверку генерации ответов на наборе из 50 вопросов.

Результаты экспериментальной оценки сохраняются в файловом контуре проекта в форматах JSONL/CSV/YAML. PostgreSQL используется для хранения корпуса, фрагментов, пользователей, запросов, ответов, цитат и обратной связи. Таблицы для хранения экспериментов в актуальной схеме БД не используются.

## Данные

Файлы эксперимента лежат в `eval/`.

- `eval/eval_questions.jsonl` - вопросы с предварительной разметкой по акту и структурной ссылке.
- `eval/gold_labels.jsonl` - эталонная разметка до сопоставления с базой.
- `eval/gold_resolved.jsonl` - эталонная разметка после сопоставления с реальными `chunk_id`.
- `eval/unresolved_gold.csv` - случаи, которые требуют ручной проверки.
- `eval/categories.md` - описание категорий вопросов.
- `eval/questions_seed.md` - человекочитаемая таблица вопросов.

Перед расчетом метрик нужно получить реальные `chunk_id` для эталонов:

```bash
.venv/bin/python scripts/eval/resolve_gold_chunks.py \
  --input eval/eval_questions.jsonl \
  --output eval/gold_resolved.jsonl \
  --unresolved eval/unresolved_gold.csv
```

## Оценка retrieval

Retrieval-эксперимент сравнивает три режима поиска:

- `sparse` - полнотекстовый поиск PostgreSQL;
- `dense` - векторный поиск по `pgvector`;
- `weighted_hybrid` - объединение sparse и dense через weighted RRF.

Для baseline-прогона использовались 50 вопросов, `top_k=10`, `candidate_limit=50`, `rrf_k=60` и три набора весов для hybrid:

- `sparse_weight=0.4`, `dense_weight=0.6`;
- `sparse_weight=0.5`, `dense_weight=0.5`;
- `sparse_weight=0.3`, `dense_weight=0.7`.

Команда запуска:

```bash
.venv/bin/python scripts/eval/run_retrieval_eval.py \
  --input eval/gold_resolved.jsonl \
  --settings settings.yaml \
  --device cpu \
  --top-k 10 \
  --candidate-limit 50 \
  --out-dir eval/results/retrieval
```

Результаты:

- `eval/results/retrieval/retrieval_runs.jsonl` - подробный JSONL по каждому вопросу и конфигурации.
- `eval/results/retrieval/summary_metrics.csv` - сводные метрики.
- `eval/results/retrieval/error_analysis.csv` - таблица ошибок для ручного анализа.
- `eval/results/retrieval/config_snapshot.yaml` - зафиксированные параметры прогона.

Основные метрики:

- `hit_at_1`, `hit_at_5`, `hit_at_10` - найден ли хотя бы один эталонный chunk в первых K результатах;
- `recall_at_5`, `recall_at_10` - доля найденных эталонных chunk в первых K результатах;
- `mrr` - reciprocal rank первого релевантного результата;
- `avg_latency_ms`, `median_latency_ms` - средняя и медианная задержка retrieval.

## Оценка генерации

Generation-эксперимент проверяет полный контур:

```text
retrieval -> token budget -> prompt answer_v1 -> GigaChat -> JSON validation -> answer_citations
```

Baseline-прогон выполнен на 50 вопросах. В сохраненном результате использовались:

- retrieval: `weighted_hybrid`;
- `top_k=10`;
- `candidate_limit=50`;
- `sparse_weight=0.5`;
- `dense_weight=0.5`;
- prompt: `answer_v1`;
- LLM provider: `gigachat`;
- модель из прогона: `GigaChat:2.0.28.2`.

Команда запуска полного прогона:

```bash
.venv/bin/python scripts/eval/run_generation_eval.py \
  --input eval/gold_resolved.jsonl \
  --settings settings.yaml \
  --device cpu \
  --limit 50 \
  --out-dir eval/results/generation
```

После запуска скрипт сохраняет итоговый JSONL как `generation_all_v1.jsonl`, а таблицу ручной оценки - как `generation_all_v1.csv`.

Результаты:

- `eval/results/generation/generation_all_v1.jsonl` - полный JSONL с ответами, использованными chunk и цитатами.
- `eval/results/generation/generation_all_v1.csv` - таблица для ручной оценки качества ответа.
- `eval/results/generation/config_snapshot.yaml` - зафиксированные параметры прогона.

Поля для ручной оценки в CSV:

- `context_sufficient_manual` - достаточно ли найденного контекста;
- `answer_supported_by_citations_manual` - подтверждается ли ответ цитатами;
- `no_extra_claims_manual` - нет ли утверждений вне контекста;
- `citation_relevance_manual` - релевантны ли использованные цитаты;
- `legal_meaning_preserved_manual` - сохранен ли правовой смысл нормы;
- `comment` - пояснение проверяющего.

## Что хранить в git

В git стоит хранить:

- финальные или baseline-прогоны на 50 вопросов;
- сводные метрики;
- таблицы ручного анализа;
- snapshot-конфиги экспериментов;
- исходные вопросы и gold-разметку.

Не стоит хранить:

- временные debug-ответы LLM;
- `logs/llm_debug/*`;
- локальные служебные файлы вроде `.DS_Store`;
- короткие пробные прогоны, которые не используются в ВКР.

## Ограничение разметки

Предварительная разметка задана по акту и структурной ссылке. Для расчета метрик используются реальные `chunk_id`, полученные после запуска `resolve_gold_chunks.py` на текущей базе. Если одна статья разбита на несколько фрагментов, эталон может содержать несколько `chunk_id`; такие случаи нужно проверять по `unresolved_gold.csv` и при необходимости уточнять вручную.
