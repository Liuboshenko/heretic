# HOW TO WORK — Heretic (внутреннее устройство / internals)

> Технический разбор: где в проекте определяются две группы промптов и где происходит
> **перехват активаций** (residual stream) для «отказных» и «безотказных» промптов.
>
> Документ состоит из двух частей: сначала **на русском языке**, затем **на английском**.
>
> Все ссылки на код указаны в формате `файл:строка` и соответствуют текущему состоянию репозитория.

---

# 🇷🇺 Часть 1. Русская версия

## 0. ⚠️ ВАЖНО: в проекте НЕТ датасета — его нужно выгрузить из сети

**В репозитории отсутствуют какие-либо файлы данных (датасеты).** В нём есть только исходный код,
конфигурационные файлы, документация и lock-файл. Проверка состава репозитория подтверждает: нет ни
одного `.csv`, `.json`, `.jsonl`, `.parquet`, `.txt`, `.arrow` с промптами.

Поэтому **для работы проекта датасеты обязательны, и по умолчанию они скачиваются из интернета**
с Hugging Face Hub при первом запуске:

| Группа | Датасет по умолчанию (скачивается из сети) |
| :--- | :--- |
| 🟢 «безобидные» промпты (без отказов) | `mlabonne/harmless_alpaca` |
| 🔴 «вредные» промпты (с отказами)     | `mlabonne/harmful_behaviors` |

Механизм скачивания: функция `load_prompts()` вызывает `load_dataset()` из библиотеки Hugging Face
`datasets`, если указан HF-ID датасета (`src/heretic/utils.py:250-255`).

**Из этого следует:**
- Для запуска с настройками по умолчанию **нужен доступ в интернет** (хотя бы для первого скачивания;
  далее данные кэшируются локально Hugging Face).
- Если интернета нет или нужен свой набор промптов — можно **подставить локальные данные** вместо
  скачивания (см. раздел 6).

## 1. Терминология проекта

| Термин в коде | Что означает |
| :--- | :--- |
| **residuals** | Перехваченные активации — скрытые состояния (hidden states) остаточного потока после каждого слоя |
| **good / "harmless"** | 🟢 Позитивные промпты, на которых модель обычно НЕ отказывает |
| **bad / "harmful"** | 🔴 Промпты, на которых модель обычно отказывает |
| **refusal direction** | «Направление отказа» — вектор в пространстве активаций, вычисляемый как разница средних |

## 2. Где определены две группы промптов

### Определения по умолчанию — `src/heretic/config.py`

Заданы четыре набора (по два на группу: для расчёта направлений и для оценки качества):

| Поле в Settings | Группа | Датасет по умолчанию | Split | Строка |
| :--- | :--- | :--- | :--- | :--- |
| `good_prompts` | 🟢 без отказов | `mlabonne/harmless_alpaca` | `train[:400]` | `config.py:475` |
| `bad_prompts` | 🔴 с отказами | `mlabonne/harmful_behaviors` | `train[:400]` | `config.py:486` |
| `good_evaluation_prompts` | 🟢 для оценки | `mlabonne/harmless_alpaca` | `test[:100]` | `config.py:497` |
| `bad_evaluation_prompts` | 🔴 для оценки | `mlabonne/harmful_behaviors` | `test[:100]` | `config.py:506` |

- Структура каждого набора описана классом `DatasetSpecification` — `config.py:40`
  (поля `dataset`, `split`, `column`, `prefix`, `suffix`, `system_prompt`).
- Те же значения продублированы в редактируемом шаблоне — `config.default.toml:178-204`
  (секции `[good_prompts]`, `[bad_prompts]`, `[good_evaluation_prompts]`, `[bad_evaluation_prompts]`).

### Загрузка промптов в память

- Функция `load_prompts()` — `src/heretic/utils.py:224`. Читает HF-датасет / локальную папку /
  текстовый файл и оборачивает каждый промпт в `Prompt(system, user)` (`utils.py:201`).
- Вызовы для основных групп (в `run()`):
  - 🟢 `good_prompts = load_prompts(settings, settings.good_prompts)` — `main.py:389`
  - 🔴 `bad_prompts  = load_prompts(settings, settings.bad_prompts)`  — `main.py:394`
- Evaluation-группы загружаются в конструкторе `Evaluator` — `evaluator.py:28` и `evaluator.py:38`.

## 3. Где перехватываются активации (residuals)

### 🎯 Само место перехвата — `Model.get_residuals()` в `src/heretic/model.py:697`

Это единственная точка, где реально снимаются активации. Механизм — не ручной forward-hook,
а встроенный флаг `output_hidden_states=True` у `model.generate()`:

```python
# model.py:700 — генерируем ровно 1 токен и просим вернуть hidden states
_, outputs = self.generate(
    prompts,
    max_new_tokens=1,
    output_hidden_states=True,      # ← ПЕРЕХВАТ АКТИВАЦИЙ
    return_dict_in_generate=True,
    use_cache=False,
)
hidden_states = outputs.hidden_states[0]     # model.py:716 — состояния для 1-го ген. токена

# model.py:719 — берём вектор на ПОСЛЕДНЕЙ позиции промпта, стек по слоям
residuals = torch.stack(
    [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
    dim=1,
)   # итоговая форма (prompt, layer, hidden_dim)
```

Дальше в той же функции: приведение к `float32`, опциональная винзоризация (`model.py:731`),
выгрузка на CPU для экономии VRAM. Обёртки над `get_residuals()`:

- `get_residuals_batched()` — `model.py:749` (батчами, полный тензор всех residuals);
- `get_residuals_mean()` — `model.py:757` (потоковое среднее, экономит VRAM).

### Где перехват вызывается для обеих групп — `src/heretic/main.py:499-528`

Есть два пути (зависят от того, нужны ли полные residuals для research-анализа):

```python
# main.py:503 — путь A (research: --print-residual-geometry / --plot-residuals)
good_residuals = model.get_residuals_batched(good_prompts)   # 🟢
bad_residuals  = model.get_residuals_batched(bad_prompts)    # 🔴
good_means = good_residuals.mean(dim=0)
bad_means  = bad_residuals.mean(dim=0)

# main.py:523 — путь B (по умолчанию: нужны только средние)
good_means = model.get_residuals_mean(good_prompts)          # 🟢
bad_means  = model.get_residuals_mean(bad_prompts)           # 🔴
```

## 4. Как из активаций строится «направление отказа» — `src/heretic/main.py:528`

```python
# refusal direction = разница средних (difference-of-means): «плохие» минус «хорошие»
refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)
```

Опционально — ортогонализация относительно «хорошего» направления (`main.py:530-540`).
Затем это направление удаляется из весов модели в `Model.abliterate()` — `model.py:461`.

## 5. Итоговая схема потока данных

```
config.py (good_prompts / bad_prompts)             ← ГДЕ ОПРЕДЕЛЕНЫ группы (только ссылки на датасеты!)
config.default.toml [good_prompts]/[bad_prompts]      (редактируемый дубль)
        │
        ▼  load_prompts()  (utils.py:224)  →  load_dataset() СКАЧИВАЕТ данные из HF (utils.py:250)
        │
main.py:389/394   good_prompts / bad_prompts  →  list[Prompt(system, user)]
        │
        ▼  get_residuals_mean() / get_residuals_batched()
        │
model.py:697  get_residuals()  ← 🎯 ПЕРЕХВАТ АКТИВАЦИЙ
        │        (generate + output_hidden_states=True; вектор на последней позиции промпта)
        ▼
main.py:509/524   good_means (🟢)     bad_means (🔴)
        │
        ▼
main.py:528   refusal_directions = normalize(bad_means − good_means)
        │
        ▼
model.py:461  abliterate()  — удаление этого направления из весов через LoRA
```

## 6. Как подставить свои промпты (без скачивания из сети)

В поле `dataset` (в `config.toml` или через CLI) вместо HF-ID можно указать:

- **путь к текстовому файлу** — один промпт на строку, пустые строки игнорируются
  (`utils.py:231-241`); поле `column` при этом не нужно;
- **путь к локальной папке датасета** (сохранённого через `datasets.save_to_disk`) — `utils.py:256`;
- **путь к локальной директории с датасетом** в стандартном формате — `utils.py:266`.

Пример `config.toml` с локальными файлами:

```toml
[good_prompts]
dataset = "./my_harmless_prompts.txt"

[bad_prompts]
dataset = "./my_harmful_prompts.txt"
```

## 7. Отдельно: как считаются отказы при оценке (другой механизм)

Важно не путать: на этапе **оценки** (`Evaluator`) отказы НЕ определяются через активации.
Ответ модели генерируется как текст, и отказ детектируется **по наличию текстовых маркеров**
(`"sorry"`, `"i cannot"`, `"illegal"` и т.д.):

- список маркеров — `config.py:431` (`refusal_markers`);
- логика проверки — `evaluator.py:47` (`is_refusal()`), подсчёт — `evaluator.py:67` (`count_refusals()`).

Это отдельный механизм для метрики качества, а не для вычисления «направления отказа».

---

# 🇬🇧 Part 2. English version

## 0. ⚠️ IMPORTANT: the project contains NO dataset — it must be downloaded from the network

**The repository does not contain any data files (datasets).** It only ships source code,
configuration files, documentation, and a lock file. Inspecting the repository confirms there is not a
single `.csv`, `.json`, `.jsonl`, `.parquet`, `.txt`, or `.arrow` file with prompts.

Therefore **datasets are mandatory for the project to work, and by default they are downloaded from the
internet** (Hugging Face Hub) on first run:

| Group | Default dataset (downloaded from the network) |
| :--- | :--- |
| 🟢 "harmless" prompts (no refusals) | `mlabonne/harmless_alpaca` |
| 🔴 "harmful" prompts (refusals)     | `mlabonne/harmful_behaviors` |

Download mechanism: `load_prompts()` calls `load_dataset()` from the Hugging Face `datasets` library
when a Hugging Face dataset ID is given (`src/heretic/utils.py:250-255`).

**Implications:**
- Running with the default settings **requires internet access** (at least for the first download;
  afterwards the data is cached locally by Hugging Face).
- If you have no internet or want your own prompts, you can **supply local data** instead of
  downloading (see section 6).

## 1. Project terminology

| Term in the code | Meaning |
| :--- | :--- |
| **residuals** | The captured activations — hidden states of the residual stream after each layer |
| **good / "harmless"** | 🟢 Positive prompts on which the model usually does NOT refuse |
| **bad / "harmful"** | 🔴 Prompts on which the model usually refuses |
| **refusal direction** | A vector in activation space computed as a difference of means |

## 2. Where the two prompt groups are defined

### Default definitions — `src/heretic/config.py`

Four sets are defined (two per group: for computing directions and for quality evaluation):

| Field in Settings | Group | Default dataset | Split | Line |
| :--- | :--- | :--- | :--- | :--- |
| `good_prompts` | 🟢 no refusals | `mlabonne/harmless_alpaca` | `train[:400]` | `config.py:475` |
| `bad_prompts` | 🔴 refusals | `mlabonne/harmful_behaviors` | `train[:400]` | `config.py:486` |
| `good_evaluation_prompts` | 🟢 for evaluation | `mlabonne/harmless_alpaca` | `test[:100]` | `config.py:497` |
| `bad_evaluation_prompts` | 🔴 for evaluation | `mlabonne/harmful_behaviors` | `test[:100]` | `config.py:506` |

- The structure of each set is described by the `DatasetSpecification` class — `config.py:40`
  (fields `dataset`, `split`, `column`, `prefix`, `suffix`, `system_prompt`).
- The same values are duplicated in the editable template — `config.default.toml:178-204`
  (sections `[good_prompts]`, `[bad_prompts]`, `[good_evaluation_prompts]`, `[bad_evaluation_prompts]`).

### Loading prompts into memory

- The `load_prompts()` function — `src/heretic/utils.py:224`. Reads an HF dataset / local folder /
  text file and wraps each prompt into `Prompt(system, user)` (`utils.py:201`).
- Calls for the main groups (in `run()`):
  - 🟢 `good_prompts = load_prompts(settings, settings.good_prompts)` — `main.py:389`
  - 🔴 `bad_prompts  = load_prompts(settings, settings.bad_prompts)`  — `main.py:394`
- The evaluation groups are loaded in the `Evaluator` constructor — `evaluator.py:28` and `evaluator.py:38`.

## 3. Where the activations (residuals) are captured

### 🎯 The capture site — `Model.get_residuals()` in `src/heretic/model.py:697`

This is the only place where activations are actually captured. The mechanism is not a manual
forward hook, but the built-in `output_hidden_states=True` flag of `model.generate()`:

```python
# model.py:700 — generate exactly 1 token and ask for hidden states
_, outputs = self.generate(
    prompts,
    max_new_tokens=1,
    output_hidden_states=True,      # ← ACTIVATION CAPTURE
    return_dict_in_generate=True,
    use_cache=False,
)
hidden_states = outputs.hidden_states[0]     # model.py:716 — states for the 1st generated token

# model.py:719 — take the vector at the LAST prompt position, stack over layers
residuals = torch.stack(
    [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
    dim=1,
)   # final shape (prompt, layer, hidden_dim)
```

Later in the same function: cast to `float32`, optional winsorization (`model.py:731`),
offload to CPU to save VRAM. Wrappers around `get_residuals()`:

- `get_residuals_batched()` — `model.py:749` (batched, the full tensor of all residuals);
- `get_residuals_mean()` — `model.py:757` (streaming mean, saves VRAM).

### Where the capture is called for both groups — `src/heretic/main.py:499-528`

There are two paths (depending on whether full residuals are needed for research analysis):

```python
# main.py:503 — path A (research: --print-residual-geometry / --plot-residuals)
good_residuals = model.get_residuals_batched(good_prompts)   # 🟢
bad_residuals  = model.get_residuals_batched(bad_prompts)    # 🔴
good_means = good_residuals.mean(dim=0)
bad_means  = bad_residuals.mean(dim=0)

# main.py:523 — path B (default: only the means are needed)
good_means = model.get_residuals_mean(good_prompts)          # 🟢
bad_means  = model.get_residuals_mean(bad_prompts)           # 🔴
```

## 4. How the "refusal direction" is built from the activations — `src/heretic/main.py:528`

```python
# refusal direction = difference of means: "bad" minus "good"
refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)
```

Optionally — orthogonalization against the "good" direction (`main.py:530-540`).
This direction is then removed from the model weights in `Model.abliterate()` — `model.py:461`.

## 5. Overall data flow

```
config.py (good_prompts / bad_prompts)             ← WHERE the groups are DEFINED (dataset references only!)
config.default.toml [good_prompts]/[bad_prompts]      (editable duplicate)
        │
        ▼  load_prompts()  (utils.py:224)  →  load_dataset() DOWNLOADS data from HF (utils.py:250)
        │
main.py:389/394   good_prompts / bad_prompts  →  list[Prompt(system, user)]
        │
        ▼  get_residuals_mean() / get_residuals_batched()
        │
model.py:697  get_residuals()  ← 🎯 ACTIVATION CAPTURE
        │        (generate + output_hidden_states=True; vector at the last prompt position)
        ▼
main.py:509/524   good_means (🟢)     bad_means (🔴)
        │
        ▼
main.py:528   refusal_directions = normalize(bad_means − good_means)
        │
        ▼
model.py:461  abliterate()  — remove this direction from the weights via LoRA
```

## 6. How to supply your own prompts (without downloading from the network)

In the `dataset` field (in `config.toml` or via CLI) instead of an HF ID you can pass:

- **a path to a text file** — one prompt per line, empty lines ignored (`utils.py:231-241`);
  the `column` field is not needed in this case;
- **a path to a locally saved dataset folder** (saved via `datasets.save_to_disk`) — `utils.py:256`;
- **a path to a local dataset directory** in the standard format — `utils.py:266`.

Example `config.toml` with local files:

```toml
[good_prompts]
dataset = "./my_harmless_prompts.txt"

[bad_prompts]
dataset = "./my_harmful_prompts.txt"
```

## 7. Aside: how refusals are counted during evaluation (a different mechanism)

Do not confuse this: during **evaluation** (`Evaluator`), refusals are NOT detected via activations.
The model's answer is generated as text, and a refusal is detected **by the presence of text markers**
(`"sorry"`, `"i cannot"`, `"illegal"`, etc.):

- the marker list — `config.py:431` (`refusal_markers`);
- the check logic — `evaluator.py:47` (`is_refusal()`), counting — `evaluator.py:67` (`count_refusals()`).

This is a separate mechanism for the quality metric, not for computing the "refusal direction".
