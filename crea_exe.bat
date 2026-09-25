@echo off
rem ===========================================================================
rem  Costruisce GenWorld.exe con PyInstaller (sperimentale).
rem
rem  Un .exe si puo' costruire solo SU Windows, quindi questo file lo fa girare
rem  sul tuo computer: non e' un eseguibile gia' pronto, e' la ricetta.
rem
rem  Avvertenza onesta: amulet porta con se' PyMCTranslate, che e' un mucchio
rem  di file JSON, e PyInstaller di suo li lascia indietro. I --collect-all
rem  qui sotto servono a quello, ma il risultato va provato. Se l'exe non
rem  parte, GenWorld.bat resta la strada che funziona sempre.
rem ===========================================================================
setlocal
title GenWorld - costruzione eseguibile
cd /d "%~dp0"

if "%GENWORLD_VENV%"=="" set "GENWORLD_VENV=C:\venvs\genworld"
set "PY=%GENWORLD_VENV%\Scripts\python.exe"

if not exist "%PY%" (
  echo Ambiente non trovato. Lancia prima GenWorld.bat, che lo crea.
  pause
  exit /b 1
)

"%PY%" -m pip install --upgrade pyinstaller
if errorlevel 1 goto :fine

"%PY%" -m PyInstaller ^
  --noconfirm --clean --onedir --windowed ^
  --name GenWorld ^
  --collect-all PyMCTranslate ^
  --collect-all amulet ^
  --collect-all amulet_nbt ^
  --hidden-import scipy.ndimage ^
  avvia_gui.py

echo.
echo Se e' andata bene, l'eseguibile e' in dist\GenWorld\GenWorld.exe
echo (--onedir e non --onefile: un solo file impiegherebbe mezzo minuto a
echo  ogni avvio per scompattare i dati di PyMCTranslate.)
:fine
echo.
pause
