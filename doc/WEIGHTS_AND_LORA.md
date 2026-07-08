# How Heretic Changes the Weights (Without Breaking the Model)

> 🌐 **Language / Язык:** [Русский](#часть-1--русская-версия) · [English](#part-2--english-version)

> A teaching-oriented deep dive into one specific, easily-misunderstood question:
> **"When Heretic ablates a model, does it edit the weight coefficients of the layers,
> or not?"** The short answer is *"it depends on which of three kinds of weights you
> mean, and on which stage you're at."* This document explains that carefully —
> simple in spirit, rigorous in the math — for a reader who wants the idea to truly
> click.
>
> Bilingual: Russian first, then English.

---

# Часть 1 — Русская версия

## 0. Главный вопрос и главная метафора

Вопрос звучит так: *«когда мы обрабатываем модель, мы правим весовые коэффициенты в
слоях или нет?»*

Быстрая метафора, которую стоит держать в голове всё чтение:

> Heretic не «стирает и переписывает `W` карандашом по месту».
> Он **кладёт поверх листа с весами прозрачную наклейку-поправку**. Модель смотрит
> сквозь лист + наклейку и видит уже исправленную картину. Оригинальный лист под
> наклейкой не тронут. И только в самом конце, *если ты захочешь*, наклейку можно
> **впечатать** в лист насовсем.

Чтобы этот образ стал точным, введём три разных понятия «весов». Их путаница — источник
90% недопонимания.

## 1. Три вида весов — держи их раздельно

| # | Название | Что это | Меняется ли во время работы? |
| :-- | :--- | :--- | :--- |
| 1 | **Базовые веса** `W` | Исходные матрицы модели (то, что скачалось с Hugging Face). | ❌ **Нет.** Во время всего поиска они заморожены и неприкосновенны. |
| 2 | **Эффективные веса** `W_eff` | То, что *реально* применяется к активациям = `W + ΔW`, где `ΔW` — LoRA-поправка. | ✅ **Да.** Именно они и определяют новое поведение модели. |
| 3 | **Сохранённые веса** | То, что ты записываешь на диск в конце: либо `W` + отдельный адаптер, либо слитые `W' = W + ΔW`. | Зависит от выбранной стратегии экспорта (см. раздел 6). |

Твоя фраза «мы не поправляем весовые коэффициенты» — **верна для №1** и **неверна для
№2**. Разберём, почему.

## 2. Куда вообще прикладывается поправка (короткий recap)

В каждом слое трансформера в «остаточный поток» (бегущий вектор скрытого состояния)
пишут две проекции: выход внимания `o_proj` и выход MLP `down_proj`. Именно их Heretic и
правит. «Направление отказа» `v` — это единичный вектор в пространстве скрытого
состояния, вычисленный один раз как нормированная разница средних активаций (вредные −
безобидные). Подробнее — в [ARCHITECTURE_RU.md](ARCHITECTURE_RU.md), разделы 5–6.

Наша цель для каждой такой матрицы `W`: сделать так, чтобы её **выход больше не содержал
компоненты вдоль `v`**.

## 3. Математика: что значит «убрать направление» из выхода слоя

Слой считает `y = W·x`. Мы хотим убрать из `y` его проекцию на `v`. Проекция вектора `y`
на единичный `v` равна `(vᵀy)·v`. Значит «очищенный» выход:

```
y' = y − λ·(vᵀy)·v
```

- при `λ = 1` компонента вдоль `v` убирается **полностью**;
- при `0 < λ < 1` она **ослабляется** (частичная абляция — так работает весовое ядро,
  где λ = weight слоя).

Это можно записать сразу на уровне матрицы. Подставим `y = Wx`:

```
y' = Wx − λ·v·(vᵀWx) = (I − λ·v·vᵀ)·W·x
```

То есть новая, «исправленная» матрица слоя — это

```
W' = (I − λ·v·vᵀ)·W
```

Оператор `P = I − λ·v·vᵀ` называется (при λ=1) **проектором**: он геометрически
«сплющивает» пространство, убирая одно направление `v`. После умножения на `W` слой
физически теряет способность выдавать `v`-компоненту.

## 4. Почему LoRA, а не прямое `W → W'`

Казалось бы: посчитали `W'`, взяли и переписали `W`. Зачем сложности? Причина в том, что
`W'` можно записать как **исходная матрица плюс маленькая добавка низкого ранга**:

```
W' = W + ΔW,   где   ΔW = −λ·v·(vᵀW)
```

Заметь структуру `ΔW`: это внешнее произведение столбца на строку — **матрица ранга 1**.
Её можно хранить не как большую матрицу, а как два маленьких вектора. Ровно это и есть
LoRA:

```
ΔW = B · A,   где   B = −λ·v   (столбец, d_out × 1)
                    A =  vᵀW   (строка,  1 × d_in)
```

Слой тогда считает `y = W·x + B·(A·x) = (W + B·A)·x`. Базовая `W` не трогается —
добавка `B·A` живёт в отдельных тензорах адаптера ([model.py:609-612](src/heretic/model.py#L609-L612)).

Что это даёт инженерно — три больших плюса:

1. **Мгновенный сброс.** Чтобы вернуть модель в исходное состояние, достаточно обнулить
   `B` (→ `B·A = 0`). Модель снова бит-в-бит оригинальная. Не нужно перезагружать веса с
   диска ([model.py:333](src/heretic/model.py#L333)).
2. **Дёшево по памяти.** Хранить `B` и `A` — это два вектора, а не полная матрица `d×d`.
3. **Обратимость и «эталон под рукой».** Оригинал всегда доступен — а он нужен, чтобы
   сравнивать с ним (см. KL в разделе 5).

> Технический нюанс: при режиме `row_normalization = "full"` (по умолчанию) добавка не
> строго ранга 1, а ранга `r` (по умолчанию 3) — это norm-preserving вариант, где `ΔW`
> приближается низкоранговым SVD. Суть та же: `W' = W + B·A`, просто `B`, `A` шире.

## 5. Цикл поиска: что мы храним и как понимаем «сломали / не сломали»

Здесь уточним твоё понимание — оно правильное по сути, но с важной поправкой про то,
*что именно* хранится.

Как ты верно заметил, поиск может быть долгим: по умолчанию **200 попыток (trials)**.
Каждая попытка — это свой набор параметров абляции (форма ядра, сила `λ` по слоям, индекс
направления). Для каждой попытки надо понять: «а не сломали ли мы модель?»

### 5.1 Как измеряется «сломали ли»: KL-дивергенция

Перед циклом мы **один раз** снимаем «эталон» — распределения первого токена **исходной**
модели на безобидных промптах (`base_logprobs`). Затем в каждой попытке снимаем те же
распределения у *обработанной* модели и считаем **KL-дивергенцию** между ними
([evaluator.py:98](src/heretic/evaluator.py#L98)):

- KL ≈ 0 → обработанная модель отвечает почти как оригинал → **интеллект сохранён**;
- KL большая → модель «поехала» → **мы её сломали**.

Параллельно считаем число отказов на вредных промптах. Итог попытки — пара чисел
`(KL, отказы)`.

### 5.2 Что именно хранится между попытками (важная поправка)

Ты сказал: *«нам нужно хранить состояния, которые мы уже нашли, для сравнения с
наилучшим KL»*. Верно — **но мы храним не веса, а параметры и метрики.**

Хранить 200 полных копий весов модели было бы гигантским расточительством (десятки-сотни
ГБ). Вместо этого в чекпоинт ([JournalStorage `.jsonl`](src/heretic/main.py#L298))
записываются только:

- **параметры** попытки (несколько чисел: `direction_index`, `max_weight`,
  `max_weight_position`, `min_weight`, `min_weight_distance` на компонент), и
- её **метрики** (`KL`, число отказов).

Почему этого достаточно? Потому что **веса детерминированно восстанавливаются из
параметров**: `reset_model()` (обнулить адаптер) → `abliterate(параметры)` (пересчитать
`B·A`). То есть любую найденную «хорошую» точку мы можем воссоздать за секунды, не храня
её веса. Оптимизатор (Optuna TPE) на основе этой истории `(параметры → метрики)` умно
выбирает, что пробовать дальше.

```
                 хранится в чекпоинте (легко):
   trial 1:  params₁  →  (KL=0.12, refusals=40)
   trial 2:  params₂  →  (KL=0.03, refusals=8)     ← пока лучший
   trial 3:  params₃  →  (KL=0.55, refusals=2)     ← мало отказов, но модель сломана
   ...
   trial 200: params₂₀₀ → (...)

   НЕ хранится: сами матрицы весов каждой попытки. Они восстанавливаются
   из params по требованию: reset_model() + abliterate(params).
```

### 5.3 «А если хорошей точки так и не нашлось?»

Тоже верно подмечено. Оптимизатор строит **фронт Парето** — набор лучших компромиссов
между «мало отказов» и «низкая KL». Если даже лучшая точка на фронте имеет **высокую KL**
(эмпирическое правило: KL > 0.5 — уже заметный урон, [main.py:776](src/heretic/main.py#L776)
предупреждает об этом), значит для этой модели/настроек снять цензуру без поломки не
удалось — и такую модель лучше **не использовать**. Ты выбираешь точку из фронта сам,
видя обе метрики.

## 6. Финал: когда новые веса пишутся на место старых

И вот теперь — прямой ответ на твою мысль: *«после того как убедились, что ничего не
сломали, можно записать новые веса на место старых»*. Да — но это **отдельный,
осознанный шаг на экспорте**, а не автоматом.

Выбрав хорошую попытку, ты решаешь, как сохранить модель
([main.py:101](src/heretic/main.py#L101)):

| Стратегия | Что происходит с весами | Когда удобно |
| :--- | :--- | :--- |
| **`adapter`** | `W` остаются нетронутыми. Сохраняется только маленький LoRA-адаптер (`B`, `A`). Модель = оригинал + адаптер. | Хочешь лёгкий файл, гибкость, возможность влить позже. |
| **`merge`** | `merge_and_unload()` **впечатывает** добавку в веса: `W' = W + B·A`, адаптер исчезает ([model.py:266](src/heretic/model.py#L266), [model.py:309](src/heretic/model.py#L309)). | Хочешь самостоятельную модель без адаптера («по классике»). |

Именно на шаге **merge** «новые эффективные веса записываются на место старых» —
физически, в сохранённой на диск модели. До этого момента базовые `W` всегда оставались
целыми.

## 7. Числовой пример (чтобы «щёлкнуло»)

Возьмём игрушечный слой в 2D. Пусть «направление отказа» — это вторая координата:

```
v = [0, 1]            (единичный вектор)
W = [[2, 1],
     [3, 4]]          (веса слоя)
x = [1, 1]            (вход)
```

**Оригинальный выход:** `y = W·x = [2·1 + 1·1,  3·1 + 4·1] = [3, 7]`.
Компонента вдоль `v` (то самое «направление отказа») = вторая координата = `7`.

**Строим поправку (λ = 1):**

```
B = −λ·v = [0, −1]              (столбец)
A = vᵀW  = [3, 4]              (строка = вторая строка W)
ΔW = B·A = [[ 0,  0],
            [−3, −4]]
```

**Эффективная матрица:**

```
W' = W + ΔW = [[2, 1],
               [0, 0]]
```

**Новый выход:** `y' = W'·x = [3, 0]`. 

Компонента вдоль `v` стала нулём — модель больше **физически не может** выдать «отказ» из
этого слоя. При этом:

- Базовая `W` в памяти **не изменилась** — изменение живёт в `ΔW = B·A` (два маленьких
  вектора `B`, `A`).
- Хотим ослабить, а не убрать? Берём `λ = 0.5`: тогда `y' = [3, 3.5]` — компонента вдоль
  `v` уменьшена вдвое.
- Хотим сохранить насовсем? На merge пишем `W'` вместо `W`. До этого — `W` цел.

Один этот пример содержит всю суть технологии: *проекция убирает направление → поправка
хранится факторизованно как `B·A` → оригинал цел → merge впечатывает по желанию.*

## 8. Что запомнить (mental model на одну строку каждое)

1. **Базовые веса `W` во время поиска не трогаются** — они заморожены.
2. **Эффективные веса меняются** через добавку `W_eff = W + B·A` (это и есть новое
   поведение).
3. **Поправка `ΔW` = проекция, убирающая `v`**, хранится факторизованно (LoRA), поэтому
   сброс = обнулить `B`.
4. **В цикле хранятся параметры и метрики, а не веса** — веса восстанавливаются из
   параметров за секунды.
5. **KL — измеритель урона**: мал → цел, велик → сломали; выбор — по фронту Парето.
6. **Только merge на экспорте пишет `W' = W + B·A` на место `W`** физически.

---

# Part 2 — English version

## 0. The core question and the core metaphor

The question: *"when we process a model, do we edit the weight coefficients of the
layers or not?"*

Keep this metaphor in mind throughout:

> Heretic does not "erase and rewrite `W` in place with a pencil."
> It **lays a transparent correction sticker over the sheet of weights**. The model
> looks through sheet + sticker and sees the already-corrected picture. The original
> sheet underneath is untouched. Only at the very end, *if you choose*, can the sticker
> be **printed into** the sheet permanently.

To make this precise, we distinguish three kinds of "weights". Confusing them causes 90%
of the misunderstanding.

## 1. Three kinds of weights — keep them separate

| # | Name | What it is | Changed during processing? |
| :-- | :--- | :--- | :--- |
| 1 | **Base weights** `W` | The model's original matrices (what you downloaded). | ❌ **No.** Frozen and untouched throughout the search. |
| 2 | **Effective weights** `W_eff` | What is *actually* applied to activations = `W + ΔW`, where `ΔW` is the LoRA correction. | ✅ **Yes.** These define the new behavior. |
| 3 | **Exported weights** | What you write to disk at the end: either `W` + a separate adapter, or merged `W' = W + ΔW`. | Depends on the export strategy (section 6). |

Your phrasing "we don't edit the weight coefficients" is **true for #1** and **false for
#2**. Here's why.

## 2. Where the correction is applied (short recap)

In every transformer layer, two projections write into the "residual stream": the
attention output `o_proj` and the MLP output `down_proj`. Those are what Heretic edits.
The "refusal direction" `v` is a unit vector in hidden-state space, computed once as the
normalized difference of mean activations (harmful − harmless). See
[ARCHITECTURE.md](ARCHITECTURE.md), sections 5–6.

Our goal for each such matrix `W`: make its **output no longer contain any component
along `v`**.

## 3. The math: what "remove a direction" from a layer's output means

A layer computes `y = W·x`. We want to strip `y` of its projection onto `v`. The
projection of `y` onto unit `v` is `(vᵀy)·v`. So the "cleaned" output is:

```
y' = y − λ·(vᵀy)·v
```

- with `λ = 1` the component along `v` is removed **completely**;
- with `0 < λ < 1` it is **attenuated** (partial ablation — this is how the per-layer
  weight kernel works, with λ = the layer's weight).

Written at the matrix level, substituting `y = Wx`:

```
y' = Wx − λ·v·(vᵀWx) = (I − λ·v·vᵀ)·W·x
```

So the new, "corrected" layer matrix is

```
W' = (I − λ·v·vᵀ)·W
```

The operator `P = I − λ·v·vᵀ` is (for λ=1) a **projector**: geometrically it flattens the
space by removing one direction `v`. After multiplying `W`, the layer physically loses the
ability to emit the `v`-component.

## 4. Why LoRA instead of directly doing `W → W'`

You might ask: just compute `W'` and overwrite `W`. Why the complication? Because `W'` can
be written as **the original matrix plus a small low-rank correction**:

```
W' = W + ΔW,   where   ΔW = −λ·v·(vᵀW)
```

Note the structure of `ΔW`: it's an outer product of a column and a row — a **rank-1
matrix**. It can be stored not as a big matrix but as two small vectors. That is exactly
LoRA:

```
ΔW = B · A,   where   B = −λ·v   (column, d_out × 1)
                      A =  vᵀW   (row,    1 × d_in)
```

The layer then computes `y = W·x + B·(A·x) = (W + B·A)·x`. Base `W` is untouched — the
`B·A` correction lives in separate adapter tensors
([model.py:609-612](src/heretic/model.py#L609-L612)).

Three big engineering wins:

1. **Instant reset.** To restore the model, just zero `B` (→ `B·A = 0`). The model is
   bit-for-bit original again — no reloading weights from disk
   ([model.py:333](src/heretic/model.py#L333)).
2. **Cheap in memory.** Storing `B` and `A` is two vectors, not a full `d×d` matrix.
3. **Reversibility and a reference on hand.** The original is always available — and you
   need it to compare against (KL, section 5).

> Technical nuance: with `row_normalization = "full"` (the default), the correction isn't
> strictly rank-1 but rank `r` (default 3) — a norm-preserving variant where `ΔW` is
> approximated by a low-rank SVD. Same essence: `W' = W + B·A`, just wider `B`, `A`.

## 5. The search loop: what we store, and how we know "broken / not broken"

Here we refine your understanding — right in spirit, with one important correction about
*what exactly* is stored.

As you noted, the search can be long: **200 trials** by default. Each trial is a set of
ablation parameters (kernel shape, per-layer strength `λ`, direction index). For each we
must ask: "did we break the model?"

### 5.1 How "broken" is measured: KL divergence

Before the loop we capture a "reference" **once** — the first-token distributions of the
**original** model on harmless prompts (`base_logprobs`). Then in each trial we capture the
same distributions from the *processed* model and compute the **KL divergence** between
them ([evaluator.py:98](src/heretic/evaluator.py#L98)):

- KL ≈ 0 → the processed model answers almost like the original → **intelligence
  preserved**;
- KL large → the model has drifted → **we broke it**.

In parallel we count refusals on harmful prompts. A trial's result is a pair `(KL,
refusals)`.

### 5.2 What exactly is stored between trials (the important correction)

You said: *"we need to store the states we've already found, to compare against the best
KL."* Correct — **but we store parameters and metrics, not weights.**

Storing 200 full copies of the model weights would be an enormous waste (tens to hundreds
of GB). Instead, the checkpoint ([JournalStorage `.jsonl`](src/heretic/main.py#L298))
records only:

- the trial's **parameters** (a handful of numbers: `direction_index`, `max_weight`,
  `max_weight_position`, `min_weight`, `min_weight_distance` per component), and
- its **metrics** (`KL`, refusal count).

Why is that enough? Because **weights are deterministically reconstructed from the
parameters**: `reset_model()` (zero the adapter) → `abliterate(parameters)` (recompute
`B·A`). Any good point found can be recreated in seconds without storing its weights. The
optimizer (Optuna TPE) uses this `(parameters → metrics)` history to smartly pick what to
try next.

```
                 stored in the checkpoint (lightweight):
   trial 1:  params₁  →  (KL=0.12, refusals=40)
   trial 2:  params₂  →  (KL=0.03, refusals=8)     ← best so far
   trial 3:  params₃  →  (KL=0.55, refusals=2)     ← few refusals, but model broken
   ...
   trial 200: params₂₀₀ → (...)

   NOT stored: the weight matrices of each trial. They are reconstructed
   from params on demand: reset_model() + abliterate(params).
```

### 5.3 "What if no good point is ever found?"

Also a sharp observation. The optimizer builds a **Pareto front** — the set of best
trade-offs between "few refusals" and "low KL". If even the best point on the front has a
**high KL** (rule of thumb: KL > 0.5 is already noticeable damage;
[main.py:776](src/heretic/main.py#L776) warns about it), then for this model/settings
decensoring without breaking failed — and such a model is better **not used**. You pick a
point from the front yourself, seeing both metrics.

## 6. The finale: when new weights are written over the old ones

Now the direct answer to your thought: *"once we've confirmed nothing is broken, we can
write the new weights over the old ones."* Yes — but it's a **separate, deliberate step at
export**, not automatic.

Having picked a good trial, you decide how to save the model
([main.py:101](src/heretic/main.py#L101)):

| Strategy | What happens to the weights | When it's convenient |
| :--- | :--- | :--- |
| **`adapter`** | `W` stays untouched. Only the small LoRA adapter (`B`, `A`) is saved. Model = original + adapter. | You want a small file, flexibility, the option to merge later. |
| **`merge`** | `merge_and_unload()` **prints** the correction into the weights: `W' = W + B·A`, the adapter disappears ([model.py:266](src/heretic/model.py#L266), [model.py:309](src/heretic/model.py#L309)). | You want a standalone model with no adapter (the "classic" case). |

It is exactly at the **merge** step that "the new effective weights are written over the
old ones" — physically, in the model saved to disk. Until then, the base `W` always stayed
intact.

## 7. A numeric example (to make it click)

Take a toy 2D layer. Let the "refusal direction" be the second coordinate:

```
v = [0, 1]            (unit vector)
W = [[2, 1],
     [3, 4]]          (layer weights)
x = [1, 1]            (input)
```

**Original output:** `y = W·x = [2·1 + 1·1,  3·1 + 4·1] = [3, 7]`.
The component along `v` (the "refusal direction") = second coordinate = `7`.

**Build the correction (λ = 1):**

```
B = −λ·v = [0, −1]              (column)
A = vᵀW  = [3, 4]              (row = second row of W)
ΔW = B·A = [[ 0,  0],
            [−3, −4]]
```

**Effective matrix:**

```
W' = W + ΔW = [[2, 1],
               [0, 0]]
```

**New output:** `y' = W'·x = [3, 0]`. 

The component along `v` became zero — the layer **physically can no longer** emit a
"refusal". And:

- Base `W` in memory **did not change** — the change lives in `ΔW = B·A` (two small
  vectors `B`, `A`).
- Want to attenuate rather than remove? Use `λ = 0.5`: then `y' = [3, 3.5]` — the
  `v`-component is halved.
- Want to keep it permanently? At merge you write `W'` in place of `W`. Until then, `W` is
  intact.

This one example contains the whole essence: *projection removes a direction → the
correction is stored factorized as `B·A` → the original stays intact → merge prints it in
on demand.*

## 8. What to remember (one line each)

1. **Base weights `W` are not touched during search** — they are frozen.
2. **Effective weights change** via the correction `W_eff = W + B·A` (this is the new
   behavior).
3. **The correction `ΔW` = a projection removing `v`**, stored factorized (LoRA), so reset
   = zero `B`.
4. **The loop stores parameters and metrics, not weights** — weights are reconstructed
   from parameters in seconds.
5. **KL is the damage meter**: small → intact, large → broken; you choose from the Pareto
   front.
6. **Only merge at export writes `W' = W + B·A` over `W`** physically.

---

*See also: [ARCHITECTURE.md](ARCHITECTURE.md) / [ARCHITECTURE_RU.md](ARCHITECTURE_RU.md) for
the full pipeline, section 8 (the ablation math) and section 10 (scoring & Pareto selection).*
