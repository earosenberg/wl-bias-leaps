#!/bin/csh
foreach n ( 0.13)
anacondaOff
python /home/rosenberg/Documents/wl-bias-leaps-top/wl-bias-leaps/reconv_shear_metacal.py $n &
end
