#!/bin/csh
foreach l (550 800)
mkdir lambda$l
foreach n ( 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13)
mkdir lambda$l/ps$n
mv *lambda$l*pixelscale$n.*.pkl lambda$l/ps$n
end
end
