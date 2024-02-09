@echo off

REM Replace 'my_python_file.py' with the name of your Python file
set python_file=my_python_file.py

REM Get the current date and time
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set datetime=%datetime:~0,8%_%datetime:~8,6%

REM Create the output file name by appending the current date and time
set output_file=output_%datetime%.txt

REM Run the Radon commands
radon cc %python_file% > %output_file%
radon raw %python_file% >> %output_file%
radon mi %python_file% >> %output_file%