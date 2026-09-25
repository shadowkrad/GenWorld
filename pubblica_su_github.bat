@echo off
rem ===========================================================================
rem  Pubblica GenWorld su https://github.com/shadowkrad/GenWorld.git
rem
rem  Da lanciare UNA VOLTA, con doppio clic, dentro la cartella del progetto.
rem  Al primo push Windows apre una finestra per accedere a GitHub: e' normale.
rem ===========================================================================
setlocal
title GenWorld - pubblicazione su GitHub
cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
  echo Git non e' installato. Scaricalo da https://git-scm.com/download/win
  echo e rilancia questo file.
  pause
  exit /b 1
)

if not exist ".git" (
  echo Creo il repository...
  git init -b main
) else (
  echo Repository gia' presente, continuo.
)

rem il nome che comparira' sui commit
git config user.name "Alessio Guidelli"
git config user.email "guidelli.alessio@gmail.com"

git add -A
git commit -m "GenWorld: da una mappa disegnata a un mondo Minecraft" -m "Pipeline completa e provata in gioco sulla 1.21.4: importazione di mappe disegnate, terreno, erosione, fiumi, vulcani, biomi, citta' organiche con mura e mercato, botteghe e abitanti, campi e frutteti, finestra PySide6, 196 test. Le mappe usate per lo sviluppo non sono incluse: sono opere di altri."

git remote remove origin >nul 2>nul
git remote add origin https://github.com/shadowkrad/GenWorld.git
git push -u origin main

echo.
echo Se e' andata bene, il progetto e' su
echo   https://github.com/shadowkrad/GenWorld
echo.
pause
