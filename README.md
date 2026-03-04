# Project Root — Looped Transformer (PyTorch) для русского языка и кода

Этот репозиторий нужен, чтобы **быстро запустить обучение своей компактной LLM** (decoder-only, GPT-подобной),
которая умеет работать с русским текстом и кодом.

Главная особенность проекта — **Looped Reasoning**:
- один и тот же `ReasoningBlock` используется повторно `n_loops` раз,
- модель может «подумать» несколько циклов перед предсказанием,
- есть `Exit Head`, который оценивает, достаточно ли текущего уровня «стабильности» скрытого состояния.

---

## Быстрый старт (если хотите запустить за 5 минут)

### 1) Установите зависимости

```bash
pip install torch tiktoken
```

### 2) Подготовьте данные

Положите ваши файлы в папку, например `./corpus`.

Поддерживаются расширения:
- `.py`, `.txt`, `.xml`, `.json`, `.md`, `.yaml`, `.yml`, `.sql`, `.c`, `.cpp`

### 3) Запустите обучение

> По умолчанию скрипт запускается с `--device cuda` (GPU-first). Если CUDA недоступна, он автоматически переключится на CPU.

```bash
python train.py --data_path ./corpus --out_dir checkpoints
```

Принудительно указать GPU:

```bash
python train.py --data_path ./corpus --out_dir checkpoints --device cuda
```

### 4) Сгенерируйте текст

> Для инференса по умолчанию также используется `--device cuda` (GPU-first), с автопереходом на CPU при отсутствии CUDA.

```bash
python chat.py --checkpoint checkpoints/latest.pt --prompt "Привет!" --max_new_tokens 150
```

Принудительно указать GPU:

```bash
python chat.py --checkpoint checkpoints/latest.pt --prompt "Привет!" --max_new_tokens 150 --device cuda
```

---

## Что есть в проекте

- `model.py`
  - Looped Transformer архитектура,
  - `ReasoningBlock`,
  - рекуррентное обновление состояния,
  - weight tying (`lm_head.weight = transformer.wte.weight`),
  - `Exit Head` и multi-loop forward.

- `data_utils.py`
  - `BPETokenizer` на `tiktoken`,
  - сбор корпуса из файла/директории,
  - поддержка **готовой разметки** и **сырых файлов**,
  - формирование train/val и батчей.

- `train.py`
  - обучение,
  - resume из `checkpoints/latest.pt`,
  - warmup + cosine LR,
  - multi-exit loss (`per_loop_ce.mean()`),
  - KL-регуляризация распределения циклов,
  - штраф за слишком ранний выход (`early_exit_penalty`).

- `chat.py`
  - инференс из чекпоинта,
  - параметры `temperature`, `top_k`, `gate_threshold`, `max_loops`.

---

## Подготовка данных (подробно)

Проект умеет обучаться:
1. На **одном файле** (`--data_path ./data.txt`)
2. На **папке** (`--data_path ./corpus`) — рекурсивно пройдёт по вложенным папкам.

### Вариант A: данные уже размечены

Если файл уже содержит блоки вида:

```text
[FILE: example.py]
[CONTENT]
print("hello")
<|endoftext|>
```

то этот формат будет использован как есть.

### Вариант B: данные без разметки

Если файлы обычные (`.py`, `.md`, `.xml` и т.д.), система автоматически обернёт каждый файл в:

```text
[FILE: name.ext]
[CONTENT]
...
<|endoftext|>
```

Это полезно, потому что модель начинает различать тип контента (код, markdown, json...).

---

## Обучение (подробно)

## 1) Базовый запуск

```bash
python train.py --data_path ./corpus --out_dir checkpoints
```

## 2) Важные аргументы

### Данные
- `--data_path` — путь к файлу или папке с данными (**основной параметр**)
- `--text_path` — legacy-алиас (оставлен для совместимости)
- `--val_ratio` — доля валидации (по умолчанию `0.1`)
- `--encoding_name` — `cl100k_base` (рекомендуется) или `gpt2`
- `--device` — `cuda` (по умолчанию), `auto` или `cpu`
  - `cuda`: GPU-first режим с fallback на CPU, если CUDA нет
  - `auto`: выбрать CUDA при наличии, иначе CPU

### Архитектура
- `--block_size` — длина контекста
- `--n_embd` — размер эмбеддинга
- `--n_head` — число attention-голов
- `--n_layer` — глубина shared-группы в loop-блоке
- `--n_loops` — число reasoning-циклов

### Обучение и регуляризация
- `--learning_rate`, `--min_lr`
- `--warmup_iters`, `--lr_decay_iters`
- `--weight_decay`, `--grad_clip`
- `--entropy_reg_weight` — KL-регуляризация распределения циклов
- `--early_exit_penalty_weight` — штраф за ранний выход при большой ошибке

## 3) Практический пример команды

```bash
python train.py \
  --data_path ./corpus \
  --out_dir checkpoints \
  --encoding_name cl100k_base \
  --block_size 256 \
  --n_embd 384 \
  --n_head 6 \
  --n_layer 2 \
  --n_loops 4 \
  --batch_size 16 \
  --max_iters 5000 \
  --learning_rate 3e-4 \
  --min_lr 3e-5 \
  --warmup_iters 200 \
  --lr_decay_iters 5000 \
  --entropy_reg_weight 0.01 \
  --early_exit_penalty_weight 0.05
```

## 4) Resume training (автоматически)

Если есть `checkpoints/latest.pt`, обучение продолжится автоматически.

В логах вы увидите:
- `Обнаружен чекпоинт, продолжаю обучение с шага X`

Если чекпоинта нет:
- `Чекпоинт не найден, начинаю обучение с нуля`

---

## Инференс / чат

## 1) Минимальный запуск

```bash
python chat.py \
  --checkpoint checkpoints/latest.pt \
  --prompt "Объясни, что такое рекурсия" \
  --max_new_tokens 200
```

По умолчанию используется `--device cuda` (GPU-first) с fallback на CPU.

## 2) Управление качеством генерации

- `--temperature` — чем выше, тем более «креативный» ответ
- `--top_k` — ограничение по top-k токенам
- `--gate_threshold` — порог раннего выхода по reasoning-гейту
- `--max_loops` — ограничение числа циклов рассуждения

Пример для кода:

```bash
python chat.py \
  --checkpoint checkpoints/latest.pt \
  --prompt "Напиши бинарный поиск на C++" \
  --temperature 0.8 \
  --top_k 50 \
  --gate_threshold 0.9 \
  --max_loops 4 \
  --max_new_tokens 220
```

---

## Что хранится в `latest.pt`

- `model_state_dict`
- `optimizer_state_dict`
- `config` (включая `n_loops`)
- `tokenizer`
- `step`
- `args`

---

## Рекомендации по качеству

1. Для русского+кода чаще всего лучше `cl100k_base`.
2. Данные важнее гиперпараметров: чистый и разнообразный корпус даёт основной прирост.
3. Если модель рано «закрывает» reasoning, увеличьте `early_exit_penalty_weight`.
4. Если `val_loss` не падает:
   - уменьшите `learning_rate`,
   - увеличьте размер корпуса,
   - уменьшите модель для начала.
5. Если не хватает VRAM:
   - уменьшите `batch_size`,
   - уменьшите `block_size`,
   - уменьшите `n_embd`.

---

## Типичные ошибки

### `ModuleNotFoundError: No module named 'torch'`

```bash
pip install torch
```

### `ImportError: No module named tiktoken`

```bash
pip install tiktoken
```

### `CUDA запрошена, но недоступна`

- используйте `--device cpu`,
- или установите CUDA-совместимую сборку PyTorch.

---

## Лицензия

Используйте и модифицируйте проект под свои задачи.
