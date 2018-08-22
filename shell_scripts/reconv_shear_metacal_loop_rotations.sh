#!/bin/csh
foreach n ( 0 15 30 45 60 75)
anacondaOff
python /home/rosenberg/Documents/wl-bias-leaps-top/wl-bias-leaps/reconv_shear_metacal.py $n &
end
