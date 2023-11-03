@echo off
if "%1"=="" goto help

set KMP_DUPLICATE_LIB_OK=TRUE
set fname=%~n1
set ffull=%1
set res=%2
set frames=%3
set args=%4 %5 %6 %7 %8 %9
for %%q in (1 2 3 4 5 6 7 8 9 10) do shift 
set args=%args% %0 %1 %2 %3 %4 %5 %6 %7 %8 %9
for %%q in (1 2 3 4 5 6 7 8 9 10) do shift 
set args=%args% %0 %1 %2 %3 %4 %5 %6 %7 %8 %9

python src/_genSGAN3.py -m models/%ffull% -o _out/%fname% -s %res% --frames %frames% --cubic %args%
ffmpeg -y -v warning -i _out\%fname%\%%06d.jpg  -crf 16 %fname%.mp4
rmdir /s /q _out\%fname%

goto end 

:help
echo Usage: gen model X-Y frames-transit ..
echo  e.g.: gen  stylegan3-r-ffhq-1024x1024  1600-1024 100-20  -n 2-1 -at -ar -sb 0.5 -sm 0.2

:end
