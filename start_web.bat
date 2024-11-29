@echo off
echo Setting up Breakcore AI Web Generator...

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt

:: Start the web application
echo Starting the web application...
streamlit run app.py

pause
