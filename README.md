## Рекомендательная система
### Цель:
Разработать рекомендательная система. Сервис на основании признаков пользователей и постов, а также взаимодействия между ними, возвращает по запросу для юзера N постов, которые пользователю покажут в его ленте соцсети.

### Стек:
NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, CatBoost, LightGBM, FastAPI, SQLAlchemy, Pydantic

### Этапы:
1. Загрузил данные из БД.
2. Обработал данные: преобразовал форматы, поработал с пропущенными значениями, нормализовал данные.
3. Провел фича-инжиниринг, создал признаки для модели на основе доступных данных.
4. Выбрал и обучил модель машинного обучения, используя библиотеки Python.
5. Проверил качество модели с использованием метрик и на кросс-валидации.
6. Загрузил на сервер подготовленные фичи и обученную моделью.
7. Написал сервис на FastAPI со следующим алгоритмом работы:
    - запуск сервиса:
        * загрузка предобученной ранее модели;
        * загрузка подготовленных фичей;
    - работа endpoint-а:
        * фильтрация фичей пользователя по id;
        * загрузка фичей постов для юзера;
        * объединение фичей юзеров и постов через таблицу feed_action;
        * выделение фичей из даты;
        * определение для юзера вероятности лайка постов;
        * удаление лайкнутых ранее постов;
        * выдача рекомендаций топ постов.

### Полученные результаты:
Данные обработаны, фичи подготовлены, модель обучена; качество модели на кросс-валидации составляет: recall=0.78; сервис работает корректно, время обработки одного запроса не более 0,5с; hitrate@5 на 2000 запросах составляет 0,593

### Рабочие файлы:
**project.ipynb** - ноутбук с обработкой признаков и обучением модели

**service** -  папка с файлами сервиса

**endpoint.py** - файл приложения с endpoint.

**model_lgbm.pkl** - предобученная модель LightGBM

**table_user.py** - ORM таблица пользователей

**table_post.py** - ORM таблица постов социальной сети

**table_feed.py** - ORM таблица взаимодействий пользователей и постов

**database.py** - скрипт с подключение к базе данных

**schemas.py** - модели валидации pydentic

**requirements.txt** - необходимые библиотеки