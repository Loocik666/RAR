# Project Root — Looped Transformer на PyTorch (русский язык + код)

Проект реализует компактную **decoder-only Transformer** модель с концепцией **Looped Language Model (Latent Reasoning)**:

- используется **один shared `ReasoningBlock`** (рекурсивно `n_loops` раз),
- обновление состояния: `h_{i+1} = LayerNorm(h_i + Block(h_i))`,
- входные/выходные веса связаны (Weight Tying): `lm_head.weight = transformer.wte.weight`,
- добавлен **Self-Consistency Exit Head**: оценивает, насколько стабилизировался вектор `h` между итерациями.

Поддерживаются:
- обучение с нуля,
- автопродолжение обучения из `latest.pt` (resume),
- инференс на CPU/GPU,
- BPE токенизация через `tiktoken` (`cl100k_base` / `gpt2`).

---

## Что внутри

- `model.py`
  - `GPTConfig` (включая `n_loops`),
  - `ReasoningBlock` (shared-recurrent ядро),
  - рекурсивная формула обновления скрытого состояния,
  - weight tying (`transformer.wte` <-> `lm_head`),
  - `Self-Consistency Exit Head` и per-loop логиты.

- `data_utils.py`
  - `BPETokenizer` (`tiktoken`),
  - `build_dataset`, `get_batch`.

- `train.py`
  - AdamW с выборочным weight decay,
  - warmup + cosine LR,
  - **multi-exit CE loss**,
  - **KL-регуляризация** распределения глубины,
  - checkpoint/resume из `out_dir/latest.pt`.

- `chat.py`
  - генерация с параметрами `gate_threshold` и `max_loops`.

---

## Установка

Требуется Python 3.10+.

```bash
pip install torch tiktoken
```

Проверка CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Подготовка данных

1. Создайте текстовый файл `data.txt` в UTF-8.
2. Для русского + кода лучше смешанный корпус:
   - статьи/документация,
   - диалоги,
   - исходники,
   - комментарии и README.

---

## Обучение (пошагово)

### 1) Базовый запуск

```bash
python train.py --text_path data.txt --out_dir checkpoints
```

### 2) Важные параметры (Project Root ядро)

- Токенизация:
  - `--encoding_name cl100k_base` (рекомендуется),
  - `--encoding_name gpt2`.
- Архитектура:
  - `--block_size`, `--n_embd`, `--n_head`,
  - `--n_layer` — глубина **shared-группы** блоков,
  - `--n_loops` — число loop-итераций рассуждения.
- Лосс/регуляризация:
  - базовый лосс: `per_loop_ce.mean()` (модель учится быть корректной на каждом цикле),
  - `--entropy_reg_weight`,
  - `--early_exit_penalty_weight` (штраф за ранний выход при высокой ошибке).
- Обучение:
  - `--learning_rate`, `--min_lr`, `--warmup_iters`, `--lr_decay_iters`,
  - `--weight_decay`, `--grad_clip`,
  - `--max_iters`, `--eval_interval`, `--eval_iters`.

### 3) Пример запуска под looped reasoning

```bash
python train.py \
  --text_path data.txt \
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

### 4) Resume training (автоматически)

Если `checkpoints/latest.pt` существует, скрипт:
- загрузит веса модели,
- загрузит состояние оптимизатора,
- продолжит обучение с сохранённого `step`,
- возьмёт архитектурные параметры из checkpoint-конфига.

В консоли вы увидите:
- `Обнаружен чекпоинт, продолжаю обучение с шага X`,
или
- `Чекпоинт не найден, начинаю обучение с нуля`.

---

## Инференс / чат

```bash
python chat.py \
  --checkpoint checkpoints/latest.pt \
  --prompt "Объясни, как работает quicksort" \
  --max_new_tokens 200
```

Параметры генерации:
- `--temperature` — температура сэмплирования,
- `--top_k` — top-k фильтрация,
- `--gate_threshold` — порог выхода по `exit_gate` (например `0.9`),
- `--max_loops` — ограничение числа reasoning-циклов на инференсе.

Пример с контролем рассуждения:

```bash
python chat.py \
  --checkpoint checkpoints/latest.pt \
  --prompt "Напиши функцию бинарного поиска на Python" \
  --temperature 0.8 \
  --top_k 50 \
  --gate_threshold 0.9 \
  --max_loops 4 \
  --max_new_tokens 220
```

---

## Что лежит в чекпоинте `latest.pt`

- `model_state_dict`,
- `optimizer_state_dict`,
- `config` (включая `n_loops`),
- `tokenizer`,
- `step`,
- `args` запуска.

---

## Практические советы

1. Для русского и кода обычно лучше `cl100k_base`.
2. Если модель «слишком рано уверена», увеличьте `entropy_reg_weight`.
3. Для более глубокого рассуждения увеличивайте `n_loops`, но следите за скоростью.
4. Если VRAM не хватает, уменьшайте `batch_size`, `block_size` и `n_embd`.
5. Для кода часто полезно: `temperature=0.6..0.9`, `top_k=30..80`.

---

## Частые ошибки

### `ModuleNotFoundError: No module named 'torch'`

```bash
pip install torch
```

### `ImportError` для `tiktoken`

```bash
pip install tiktoken
```

### `CUDA запрошена, но недоступна`

- Используйте `--device cpu`,
- или установите CUDA-совместимую сборку PyTorch.

---

## Лицензия

Используйте и модифицируйте под свои задачи.
