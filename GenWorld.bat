@echo off
rem ===========================================================================
rem  GenWorld - avvio con doppio clic.
rem
rem  Al primo avvio crea l'ambiente e installa le dipendenze; dopo parte e
rem  basta. L'ambiente sta FUORI da questa cartella (C:\venvs\genworld) di
rem  proposito: un venv sono migliaia di file e questa cartella e' dentro
rem  Google Drive, che li sincronizzerebbe tutti.
rem
rem  LA VERSIONE DI PYTHON CONTA, ed e' il motivo per cui qui sotto c'e' una
rem  ricerca invece di un "py -3". amulet-core vuole numpy della serie 1, e
rem  l'ultimo numpy della serie 1 (1.26.4) ha le ruote gia' compilate fino a
rem  Python 3.12. Con un Python piu' nuovo pip non trova la ruota, prova a
rem  compilare numpy dai sorgenti, cerca il compilatore di Visual Studio, non
rem  lo trova e si ferma con
rem
rem      ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ...]
rem
rem  che sembra un problema di GenWorld e non lo e'. Quindi si cerca un
rem  Python 3.12 o piu' vecchio, e se non c'e' lo si dice chiaramente invece
rem  di far partire una compilazione destinata a fallire.
rem ===========================================================================
setlocal
title GenWorld
cd /d "%~dp0"

if "%GENWORLD_VENV%"=="" set "GENWORLD_VENV=C:\venvs\genworld"
set "PY=%GENWORLD_VENV%\Scripts\python.exe"

if not exist "%PY%" call :crea_ambiente
if not exist "%PY%" goto :niente_python

rem Un ambiente gia' creato con un Python troppo nuovo non si aggiusta a colpi
rem di pip: si rifa'. Costa due minuti e sta fuori dal progetto.
call :controlla_versione
if errorlevel 1 goto :rifai_ambiente

rem Le dipendenze si controllano importandole davvero: un pacchetto presente
rem ma rotto (il caso classico: numpy salito alla 2 sotto amulet) non si vede
rem guardando l'elenco di pip.
"%PY%" -c "import genworld, PySide6, amulet, scipy, PIL" >nul 2>nul
if errorlevel 1 call :installa_dipendenze

"%PY%" -c "import genworld, PySide6, amulet, scipy, PIL" >nul 2>nul
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
rem  Cerca un interprete con cui numpy 1.26 si installa senza compilare.
rem  Le versioni si provano dalla piu' comoda alla piu' vecchia.
rem
rem  Due cose qui non sono affidabili, ed entrambe hanno gia' fatto scegliere
rem  un Python sbagliato in passato:
rem
rem  - l'errorlevel di "py -3.12 ...": su alcune installazioni del py
rem    launcher, quando la versione richiesta non c'e', stampa
rem        [ERROR] No runtime installed that matches 3.12. ...
rem    ma NON imposta un errorlevel diverso da zero, e "if not errorlevel 1"
rem    leggeva quel fallimento come un successo;
rem
rem  - un confronto numerico tipo "<=" dentro un "-c" lanciato via FOR /F:
rem    quella riga passa per un secondo cmd.exe nascosto, che dentro le
rem    virgolette legge comunque "<" come un reindirizzamento e lo toglie dal
rem    comando PRIMA che arrivi a python. "sys.version_info[:2] <= (3,12)"
rem    diventava cosi' semplicemente "sys.version_info[:2]", che e' una tupla
rem    non vuota e quindi SEMPRE vera: il controllo diceva sempre "va bene",
rem    qualunque fosse la versione vera - ed e' cosi' che un Python 3.14 e'
rem    finito nel venv nonostante il controllo.
rem
rem  Quindi: niente "<", "<=", ">" o ">=" dentro un -c, e niente fiducia
rem  nell'errorlevel di py. Si legge la versione che l'interprete DICE di
rem  essere (un numero stampato, non un confronto) e si confronta come testo
rem  contro un elenco fisso, in batch, dove "<" non compare da nessuna parte.
:trova_python
set "PYBASE="
where py >nul 2>nul
if errorlevel 1 goto :prova_python_exe
call :prova_versione 3.12
if not "%PYBASE%"=="" goto :eof
call :prova_versione 3.11
if not "%PYBASE%"=="" goto :eof
call :prova_versione 3.10
if not "%PYBASE%"=="" goto :eof
call :prova_versione 3.9
if not "%PYBASE%"=="" goto :eof
:prova_python_exe
set "RISPOSTA="
for /f "delims=" %%v in ('python -c "import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))" 2^>nul') do set "RISPOSTA=%%v"
if "%RISPOSTA%"=="3.12" set "PYBASE=python"
if "%RISPOSTA%"=="3.11" set "PYBASE=python"
if "%RISPOSTA%"=="3.10" set "PYBASE=python"
if "%RISPOSTA%"=="3.9" set "PYBASE=python"
goto :eof

rem  %1 = versione da provare (es. 3.12). Imposta PYBASE solo se "py -%1"
rem  risponde DAVVERO con quella versione - letta come testo, non dedotta da
rem  un errorlevel ne' da un confronto.
:prova_versione
set "RISPOSTA="
for /f "delims=" %%v in ('py -%1 -c "import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))" 2^>nul') do set "RISPOSTA=%%v"
if "%RISPOSTA%"=="%1" set "PYBASE=py -%1"
goto :eof

rem ---------------------------------------------------------------------------
rem  0 = la versione va bene, 1 = e' troppo nuova per numpy 1.
:controlla_versione
set "RISPOSTA="
for /f "delims=" %%v in ('"%PY%" -c "import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))" 2^>nul') do set "RISPOSTA=%%v"
if "%RISPOSTA%"=="3.12" exit /b 0
if "%RISPOSTA%"=="3.11" exit /b 0
if "%RISPOSTA%"=="3.10" exit /b 0
if "%RISPOSTA%"=="3.9" exit /b 0
exit /b 1

rem ---------------------------------------------------------------------------
:crea_ambiente
echo Primo avvio: creo l'ambiente Python in %GENWORLD_VENV%
echo Ci vuole qualche minuto, una volta sola.
echo.
call :trova_python
if "%PYBASE%"=="" goto :python_troppo_nuovo
echo Uso %PYBASE%
%PYBASE% -m venv "%GENWORLD_VENV%"
exit /b 0

rem ---------------------------------------------------------------------------
:rifai_ambiente
if "%GENWORLD_RIFATTO%"=="1" goto :python_troppo_nuovo
call :trova_python
if "%PYBASE%"=="" goto :python_troppo_nuovo
echo.
echo L'ambiente in %GENWORLD_VENV% e' stato creato con un Python troppo
echo nuovo per numpy 1, che e' quello che amulet richiede. Lo rifaccio con
echo %PYBASE%.
echo.
rmdir /s /q "%GENWORLD_VENV%"
set "GENWORLD_RIFATTO=1"
call :crea_ambiente
call :installa_dipendenze
"%PY%" -c "import genworld, PySide6, amulet, scipy, PIL" >nul 2>nul
if errorlevel 1 goto :dipendenze_mancanti
echo Avvio GenWorld...
"%PY%" -m genworld.gui
if errorlevel 1 pause
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
:python_troppo_nuovo
echo.
echo Su questo computer c'e' Python, ma e' troppo nuovo.
echo.
echo   GenWorld scrive i mondi con amulet-core, che richiede numpy della
echo   serie 1. L'ultimo numpy della serie 1 ha i pacchetti gia' compilati
echo   fino a Python 3.12: con il 3.13 o il 3.14 pip prova a compilarlo dai
echo   sorgenti, cerca il compilatore di Visual Studio e si ferma.
echo.
echo COSA FARE: installa Python 3.12 da
echo   https://www.python.org/downloads/release/python-3128/
echo (in fondo alla pagina, "Windows installer (64-bit)").
echo Puoi tenere anche quello nuovo: convivono, e questo file si prende da
echo solo il 3.12 quando lo trova.
echo.
echo Poi cancella la cartella %GENWORLD_VENV% e rilancia GenWorld.bat.
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
