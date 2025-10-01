@echo off
echo Activating Python virtual environment...
call venv\Scripts\activate

echo Starting Flask server...
python app.py

echo Flask server started. Press any key to close this window.
pause