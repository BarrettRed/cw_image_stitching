## ** Сборка проекта**
1. В корне репозитория выполните команды:
    ```bash
    cmake -B build -DCMAKE_TOOLCHAIN_FILE="<путь_до_вашей_папки_vcpkg>/scripts/buildsystems/vcpkg.cmake"
    cmake --build build --config Release
    cmake --install build --config Release
    ```

2. После сборки исполняемый файл будет находиться в папке `install/bin/`.

---

## **Запуск**
 
1. Перейдите в папку `install/bin`:
    ```bash
    cd install/bin
    ```
2. Запустите приложение через командную строку:
    ```bash
    panorama.exe <image_1> <image_2>
    ```
3. В результата будет сгенерированы 8 изображений:
- Keypoints Image 1 (ORB).png
- Keypoints Image 1 (SIFT).png
- Keypoints Image 2 (ORB).png
- Keypoints Image 2 (SIFT).png
- Matches ORB.png
- Matches SIFT.png
- panorama.cpp
- panorama_ORB.png
- panorama_SIFT.png

---

## **Документация**
Документация, сгенерированная с помощью Doxygen, доступна в папке `install/docs/html`. Для просмотра откройте файл `index.html` в браузере.  

---

## **Тестирование** (опционально)
Для запуска тестов выполните:
```bash
cd bin.rel/test
ssim_test.exe <source_image> <panorama_ORB_image> <panorama_SIFT_image>
```
