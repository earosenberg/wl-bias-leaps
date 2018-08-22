#!/bin/csh
foreach n ( 5 10 15 20 30 40 60 80 100 None)
anacondaOff
python /home/rosenberg/Documents/wl-bias-leaps-top/wl-bias-leaps/reconv_shear_metacal.py $n &
end
