# ДЗ-2: воспроизведение Cut Cross-Entropy

Статья: **Cut Your Losses in Large-Vocabulary Language Models** (Wijmans et al., Apple, ICLR 2025), [arXiv:2411.09009](https://arxiv.org/abs/2411.09009).

Курс Efficient Models, ИТМО 2026. Автор: Карина Денисова.

## Что здесь лежит

| Файл | Что это |
|---|---|
| [`slides/presentation.pdf`](slides/presentation.pdf) | Презентация с объяснением статьи|
| [`report.pdf`](report.pdf) ([md](report.md)) | Отчёт о воспроизведении |
| [`src/cce.py`](src/cce.py) | Cut Cross-Entropy и обычный CE на numpy |
| [`src/lm.py`](src/lm.py) | Маленькая языковая модель для экспериментов |
| [`src/experiments.py`](src/experiments.py) | Все эксперименты, результаты пишутся в `results/` |
| [`src/plots.py`](src/plots.py) | Графики в `figures/`|


## Главное

| Утверждение статьи | Результат |
|---|---|
| CCE точно совпадает с обычным CE | ✅ до машинной точности (10⁻¹⁸ в fp64) |
| Память не зависит от размера словаря | ✅ 6 144 МБ → 180 МБ при V = 128K (34×), CCE = градиенты + один блок |
| Обучение не страдает | ✅ кривые совпадают, расхождение val-лосса ≤ 0,023 |
| Фильтрация градиента + сортировка словаря | ✅ до 70% блоков пропускается, ⚠️ эффект зависит от разреженности softmax |

GPU не использовался: всё воспроизведено на CPU (4 ядра), реализация с нуля на numpy.

## Запуск

```bash
pip install -r requirements.txt
python src/experiments.py           
python src/experiments.py --quick   
python src/plots.py                
```

Корпус `data/Isaac.Newton-Opticks.txt` — «Opticks» Ньютона из Project Gutenberg.
