# FTP photo object detector: локальная YOLO-модель

GitHub Actions workflow, который подключается к FTP/FTPS/SFTP, рекурсивно обходит папку с фото, выбирает изображения без соседнего JSON-файла с тем же именем, запускает локальное распознавание объектов через Ultralytics YOLO и загружает результат обратно рядом с исходным изображением.

OpenAI API не используется.

Пример:

```text
photo/cat.jpg      -> анализируется, создаётся photo/cat.json
photo/cat.json     -> уже есть результат, поэтому photo/cat.jpg пропускается
photo/nested/a.png -> анализируется, создаётся photo/nested/a.json
```

## Структура репозитория

```text
.github/workflows/analyze-ftp-photos.yml
scripts/analyze_ftp_photos.py
requirements.txt
.gitignore
README.md
```

## Обязательные GitHub secrets

Используются те же FTP-секреты, что и в вашем workflow деплоя:

| Secret | Назначение |
|---|---|
| `FTP_URL` | FTP host или URL. Примеры: `example.com`, `ftp://example.com`, `ftps://example.com`, `sftp://example.com` |
| `FTP_USER` | FTP/SFTP пользователь |
| `FTP_PASS` | FTP/SFTP пароль |

`OPENAI_API_KEY` больше не нужен.

## Опциональные GitHub variables

| Variable | Default | Назначение |
|---|---:|---|
| `FTP_PHOTO_DIR` | `photo` | Удалённая папка, которую нужно сканировать |
| `LOCAL_MODEL_PATH` | `yolo11n.pt` | Путь к локальному `.pt` файлу или имя стандартной YOLO-модели |
| `YOLO_CONFIDENCE` | `0.25` | Минимальная уверенность детекции |
| `YOLO_IOU` | `0.70` | IoU threshold для подавления пересекающихся bbox |
| `YOLO_IMAGE_SIZE` | `1280` | Размер инференса YOLO |
| `YOLO_DEVICE` | `cpu` | Устройство инференса: `cpu`, `0`, `cuda:0` и т. п. На GitHub-hosted runner обычно используйте `cpu` |
| `MAX_DETECTIONS` | `300` | Максимум объектов на одно изображение |
| `MAX_FILES_PER_RUN` | `0` | `0` = без лимита; иначе обработать только N файлов за запуск |
| `MAX_IMAGE_EDGE` | `1600` | Изображение уменьшается до этого максимального ребра перед инференсом |

## Как использовать свою локальную модель

Есть два режима.

### Вариант 1. Автоматическая загрузка стандартной модели

Ничего не добавляйте в репозиторий. По умолчанию используется:

```text
yolo11n.pt
```

При первом запуске `ultralytics` скачает веса модели на runner, затем workflow будет пытаться кэшировать их через `actions/cache`.

Это не внешний AI API: изображение не отправляется в OpenAI или другой vision-сервис. Но сама модель может быть скачана при первом запуске, если файла ещё нет на runner-е.

### Вариант 2. Полностью без скачивания модели во время workflow

Положите файл модели в репозиторий, например:

```text
models/yolo11n.pt
```

И задайте variable:

```text
LOCAL_MODEL_PATH=models/yolo11n.pt
```

Такой режим лучше, если нужно, чтобы Action не зависел от доступности сервера загрузки весов.

## Workflow behavior

Workflow запускается:

- вручную через **Actions -> Analyze FTP photos with local YOLO -> Run workflow**;
- автоматически каждые 30 минут через cron.

Для каждого изображения внутри `FTP_PHOTO_DIR`:

1. Проверяет, есть ли рядом JSON с тем же basename.
2. Если JSON есть, пропускает изображение.
3. Если JSON нет, скачивает изображение.
4. Исправляет EXIF-ориентацию и приводит изображение к RGB.
5. При необходимости уменьшает изображение до `MAX_IMAGE_EDGE`.
6. Запускает локальную YOLO-модель.
7. Загружает `<same-basename>.json` в ту же удалённую папку.

## Output JSON shape

```json
{
  "schema_version": "2.0",
  "source": {
    "path": "photo/example.jpg",
    "filename": "example.jpg",
    "original_bytes": 123456,
    "original_width": 4032,
    "original_height": 3024,
    "prepared_width": 1600,
    "prepared_height": 1200,
    "source_mime_guess": "image/jpeg",
    "prepared_mode": "RGB"
  },
  "analyzed_at": "2026-06-14T12:00:00+00:00",
  "model": "yolo11n.pt",
  "description": "Local YOLO detection found 2 object(s) across 2 class(es).",
  "objects": [
    {
      "label": "cat",
      "label_ru": "кошка",
      "class_id": 15,
      "count": 1,
      "confidence": 0.9321,
      "max_confidence": 0.9321,
      "bounding_boxes": [
        {
          "x_min": 0.21,
          "y_min": 0.18,
          "x_max": 0.72,
          "y_max": 0.88,
          "confidence": 0.9321
        }
      ]
    }
  ],
  "detections": [
    {
      "label": "cat",
      "label_ru": "кошка",
      "class_id": 15,
      "confidence": 0.9321,
      "bounding_box": {
        "x_min": 0.21,
        "y_min": 0.18,
        "x_max": 0.72,
        "y_max": 0.88
      }
    }
  ],
  "inference": {
    "engine": "ultralytics-yolo",
    "model_path": "yolo11n.pt",
    "confidence_threshold": 0.25,
    "iou_threshold": 0.7,
    "image_size": 1280,
    "max_detections": 300,
    "device": "cpu"
  }
}
```

Bounding boxes — нормализованные координаты от `0` до `1` относительно подготовленного изображения.

## Важные ограничения локальной YOLO-модели

- Стандартная COCO-модель знает ограниченный набор классов: люди, автомобили, животные, мебель, бытовые предметы и т. п.
- Она не делает свободное текстовое описание сцены как LLM/Vision-модель.
- Для специфических объектов нужно обучить свою YOLO-модель и указать её через `LOCAL_MODEL_PATH`.

## Notes

- Если `FTP_URL` указан без протокола, скрипт считает, что это `ftp://`.
- Для FTPS используйте `ftps://...`.
- Для SFTP используйте `sftp://...`.
- Скрипт намеренно не перезаписывает существующие JSON-файлы.
- Чтобы одна ошибка обработки падала всем workflow, задайте env/variable `FAIL_ON_ERRORS=true`.
