# RAR — Мини-GPT на PyTorch для русского языка и кода

Проект реализует компактную **decoder-only Transformer** модель (GPT-подобную), которую можно:
- обучать **с нуля** на своём тексте;
- использовать для **дообучения/продолжения** через чекпоинт;
- запускать на **CPU** и **GPU (CUDA)**.

Ключевые улучшения для русского языка и кода:
- **BPE токенизация через `tiktoken`** (`gpt2` или `cl100k_base`),
- современное внимание через **`scaled_dot_product_attention`** (ускорение на CUDA/Flash Attention),
- **Pre-Norm архитектура с RMSNorm**,
- корректная инициализация весов,
- **AdamW с выборочным weight decay**,
- **LR scheduler: warmup + cosine decay**.

---

## Структура проекта

- `model.py` — архитектура модели (`GPTConfig`, `RMSNorm`, блоки Transformer, `GPT`)
- `data_utils.py` — BPE токенизация (`tiktoken`) и подготовка батчей
- `train.py` — обучение, валидация, чекпоинты, оптимизатор и LR scheduler
- `chat.py` — инференс/генерация текста из сохранённого чекпоинта

---

## Установка зависимостей

Рекомендуется Python 3.10+.

```bash
pip install torch tiktoken
```

Проверка CUDA в Python:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Подготовка данных

1. Подготовьте текстовый файл `data.txt` в UTF-8.
2. Для лучшего результата на русском + коде используйте смешанный корпус:
   - статьи/документация,
   - диалоги,
   - код + комментарии,
   - README/техтексты.

---

## Запуск обучения (пошагово)

### 1) Базовый запуск

```bash
python train.py --text_path data.txt --out_dir checkpoints
```

По умолчанию:
- токенизация: `cl100k_base`,
- устройство: `auto` (CUDA, если доступна),
- модель: `n_embd=384`, `n_head=6`, `n_layer=6`, `block_size=256`.

### 2) Явно выбрать токенизацию

**`cl100k_base`** (рекомендуется для русского и кода):

```bash
python train.py --text_path data.txt --encoding_name cl100k_base
```

**`gpt2`** (совместимый классический BPE):

```bash
python train.py --text_path data.txt --encoding_name gpt2
```

### 3) Запустить на GPU

```bash
python train.py --text_path data.txt --device cuda
```

Если CUDA недоступна — используйте `--device cpu` или оставьте `auto`.

### 4) Пример настройки гиперпараметров

```bash
python train.py \
  --text_path data.txt \
  --out_dir checkpoints \
  --encoding_name cl100k_base \
  --block_size 256 \
  --n_embd 512 \
  --n_head 8 \
  --n_layer 8 \
  --batch_size 12 \
  --max_iters 10000 \
  --learning_rate 3e-4 \
  --min_lr 3e-5 \
  --warmup_iters 500 \
  --lr_decay_iters 10000 \
  --weight_decay 0.1
```

### 5) Что сохраняется

В `out_dir/latest.pt` сохраняются:
- веса модели,
- состояние оптимизатора,
- конфиг модели,
- токенайзер,
- шаг обучения и аргументы запуска.

---

## Инференс / чат

После обучения:

```bash
python chat.py --checkpoint checkpoints/latest.pt --prompt "Привет! Объясни, что такое attention." --max_new_tokens 200
```

Параметры генерации:
- `--temperature` (например 0.7–1.2)
- `--top_k` (например 40)

Пример:

```bash
python chat.py \
  --checkpoint checkpoints/latest.pt \
  --prompt "Напиши функцию быстрой сортировки на Python" \
  --temperature 0.8 \
  --top_k 50 \
  --max_new_tokens 250
```

---

## Рекомендации по качеству

1. **Токенизация**: `cl100k_base` обычно лучше для мультиязычности и кода.
2. **Данные важнее архитектуры**: очистите мусор и дубликаты.
3. **Начинайте с маленькой модели**, затем масштабируйте `n_layer`, `n_embd`, `block_size`.
4. **Следите за `val_loss`**: если растёт при падении `train_loss`, усиливайте регуляризацию/данные.
5. Для генерации кода обычно полезны: `temperature=0.6..0.9`, `top_k=30..80`.

---

## Типичные проблемы

### `ModuleNotFoundError: No module named 'torch'`

Установите PyTorch:

```bash
pip install torch
```

### `ImportError` про `tiktoken`

```bash
pip install tiktoken
```

### `CUDA запрошена, но недоступна`

Запустите с `--device cpu` или установите CUDA-совместимую сборку PyTorch.

---

## Лицензия

Используйте и модифицируйте под свои задачи.
