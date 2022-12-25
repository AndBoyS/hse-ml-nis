## Проект НИС MLP в ВШЭ

Участники: Бахматов Андрей, Корякин Алексей, Дмитрий Симонян, Герман Арутюнов, Сергей Хрущев, Владимир Морозов

Задача: определение наличия очков на фотографии

### Использование

#### Инференс на папке с фото:

```shell
python main.py path/to/image/folder
```

#### Запуск приложения:

Создание виртуальной среды

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Запуск

```shell
python demo.py
```

Затем откройте http://localhost:7860 в браузере. 

#### Запуск приложения докере:

```shell
docker build -t glasses . && docker run -v $(pwd)/demo_data:/app/demo_data -p 7860:7860 glasses
```

К сожалению, в докере не работает кеб-камера.
Решить проблему можно [вот так](https://medium.com/@jijupax/connect-the-webcam-to-docker-on-mac-or-windows-51d894c44468), но это муторно, легче запустить приложение первым способом.

gif-ка с примером работы приложения:

![usage](usage.gif)


### Для коллабораторов 

Фиксирование новых зависимостей (автоматически обновит requirements.txt):

```shell
pip freeze > requirements.in
pip-compile requirements.in
```