@echo off
echo Запуск локального сервера...
echo.
echo Откройте в браузере: http://localhost:8000/prof4.0.html
echo.
echo Для остановки нажмите Ctrl+C
echo.
python -m http.server 8000
pause