@echo off
rem ===========================================================================
rem  GenWorld - avvio con doppio clic.
rem
rem  Al primo avvio crea l'ambiente e installa le dipendenze; dopo parte e
rem  basta. L'ambiente sta FUORI da questa cartella (C:\venvs\genworld) di
rem  proposito: un venv sono migliaia di file e questa cartella e' dentro
rem  Google Drive, che li sincronizzerebbe tutti.
rem ===========================================================================
setlocal
title GenWorld
cd /d "%~dp0"

if "%GENWORLD_VENV%"=="" set "GENWORLD_VENV=C:\venvs\genworld"
set "PY=%GENWORLD_VENV%\Scripts\python.exe"

if not exist "%PY%" call :crea_ambiente
if not exist "%PY%" goto :niente_python

rem Le dipendenze si controllano importandole davvero: un pacchetto presente
rem ma rotto (il caso classico: numpy salito alla 2 sotto amulet) non si vede
rem guardando l'elenco di pip.
"%PY%" -c "import PySide6, amulet, scipy, PIL" >nul 2>nul
if errorlevel 1 call :installa_dipendenze

"%PY%" -c "import PySide6, amulet, scipy, PIL" >nul 2>nul
if errorlevel 1 goto :dipendenze_mancanti

"%PY%" -c "import numpy,sys; sys.exit(0 if numpy.__version__[0]=='1' else 1)" >nul 2>nul
if errorlevel 1 echo ATTENZIONE: numpy e' salito alla versione 2 e amulet non lo regge. Sistema con:  "%PY%" -m pip install "numpy<2"

echo Avvio GenWorld...
"%PY%" -m genworld.gui
if errorlevel 1 (
  echo.
  echo GenWorld si e' chiuso con un errore. Il testo sopra dice quale.
  pause
)
exit /b 0

rem ---------------------------------------------------------------------------
:crea_ambiente
echo Primo avvio: creo l'ambiente Python in %GENWORLD_VENV%
echo Ci vuole qualche minuto, una volta sola.
echo.
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -m venv "%GENWORLD_VENV%"
) else (
  python -m venv "%GENWORLD_VENV%"
)
exit /b 0

rem ---------------------------------------------------------------------------
:installa_dipendenze
echo Installo le dipendenze in %GENWORLD_VENV%
echo.
"%PY%" -m pip install --upgrade pip
rem numpy si fissa PRIMA e per primo: amulet-core richiede la serie 1, e se un
rem altro pacchetto tira su la 2 il mondo non si scrive piu'.
"%PY%" -m pip install "numpy<2"
"%PY%" -m pip install amulet-core pillow scipy PySide6-Essentials
echo.
exit /b 0

rem ---------------------------------------------------------------------------
:niente_python
echo.
echo Non trovo Python su questo computer.
echo Installalo da https://www.python.org/downloads/ ricordando di spuntare
echo "Add Python to PATH", poi rilancia questo file.
echo.
pause
exit /b 1

rem ---------------------------------------------------------------------------
:dipendenze_mancanti
echo.
echo L'installazione delle dipendenze non e' riuscita. Provale a mano:
echo   "%PY%" -m pip install "numpy^<2" amulet-core pillow scipy PySide6-Essentials
echo.
pause
exit /b 1
