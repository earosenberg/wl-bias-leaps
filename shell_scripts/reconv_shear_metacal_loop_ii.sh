#!/bin/csh
foreach n ( 0 1 2 3 )
anacondaOff
python /home/rosenberg/Documents/wl-bias-leaps-top/wl-bias-leaps/reconv_shear_metacal.py $n &
end
