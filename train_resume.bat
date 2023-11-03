@echo off
if "%1"=="" goto help

set mode=%1
set data=%2
set init=%3
set kimg=%4
set args=%5 %6 %7 %8 %9
for %%q in (1 2 3 4 5 6 7 8 9 10) do shift 
set args=%args% %0 %1 %2 %3 %4 %5 %6 %7 %8 %9
for %%q in (1 2 3 4 5 6 7 8 9 10) do shift 
set args=%args% %0 %1 %2 %3 %4 %5 %6 %7 %8 %9

rem _uptrain X data ffhq-512.pkl 2000 ...
if %mode%==2 set cfg=sg2
if %mode%==r set cfg=sg3r
if %mode%==t set cfg=sg3t

python lib/train.py --data=data/%data% --resume models/%init% --kimg %kimg%  --cfg %cfg%  %args% 
goto end

useful options:
--resume KIMG
--batch N
--aug apa|ada
--mirror 

:help
echo Usage: train_resume type dataset initmodel kimg ..
echo  e.g.: train_resume r cats.zip stylegan3-r-afhqv2-512x512.pkl 2000

:end