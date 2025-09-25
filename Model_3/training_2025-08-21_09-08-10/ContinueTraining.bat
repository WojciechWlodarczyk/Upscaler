echo off

cd /d %~dp0

REM Szukanie pierwszego pliku z rozszerzeniem .pt w bieżącym katalogu
for %%f in (*.pt) do (
    set pt_file=%%~dpfnxf
    goto :found
)

echo ERROR: Cannot find model file!
goto :eof

:found

python "%~dp0..\..\TestBat.py" "%~dp0 " "%pt_file%"
