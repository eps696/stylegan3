@echo off
if "%1"=="" goto help

rem train X data 300 ..
if %1==2 set cfg=sg2
if %1==r set cfg=sg3r
if %1==t set cfg=sg3t

python lib/train.py --data=data/%2 --kimg %3 --cfg %cfg% ^
%4 %5 %6 %7 %8 %9

goto end 

:help
echo Usage: train type dataset kimg ..
echo  e.g.: train r cats.zip 8000

:end
