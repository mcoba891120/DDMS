@echo off
for /f "skip=1 delims=" %%x in ('wmic os get localdatetime') do if not defined X set X=%%x
set date=%X:~0,4%-%X:~4,2%-%X:~6,2%-%X:~8,2%-%X:~10,2% 

set /p TASKNAME="TASKNAME: " || set TASKNAME=%date% 
set /p CONFIG="CONFIG: "
set /p UMAT="UMAT: "

	
if defined UMAT (
	call python main.py -i %CONFIG% -tn %TASKNAME% -m write
	call ABAQUS cpus=4 input=%TASKNAME%.inp job=%TASKNAME%.inp user=./core/%UMAT%.for int
) else (
	call python main.py -i %CONFIG% -tn %TASKNAME% -m write
)

call python main.py -i %CONFIG% -tn TASKNAME -m output
