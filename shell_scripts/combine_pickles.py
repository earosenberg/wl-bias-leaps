import numpy as np
import cPickle
import os
d = '.'
subdirs = [os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]
print subdirs
for subdir in subdirs:    
    fn = sorted([fil for fil in os.listdir(subdir) if fil.endswith("pkl")])
    res = []
    nrot = len(fn)
    for filename in fn:
        fil = open(subdir+'/'+filename)
        try:
            resmetacal = cPickle.load(fil)
            ident = resmetacal[-1]
            ngal = len(ident) / nrot
            identList = [ident[i*ngal:(i+1)*ngal] for i in range(nrot)]
            assert np.all(identList == np.roll(identList, 1, axis=0))

            res.append(resmetacal)
        except:
            print fil
        fil.close()
    print nrot
    res = np.stack(res).transpose()
    #res = [np.stack(x, axis=-1) for x in res]
    rece1, rece2, shearList, hlr, sn, q, phiList, ident = res
    hlr, sn, q, phiList, ident = [np.concatenate(x, axis=1) for x in [hlr, sn, q, phiList, ident]]
    rece1, rece2 = [np.concatenate(x, axis=2) for x in [rece1, rece2]]
    assert np.all([shearList[0] == shearList[i] for i in range(len(shearList))])
    shearList = shearList[0]

    newres = [rece1, rece2, shearList, hlr, sn, q, phiList, ident]
    fn0 = fn[0]
    newname = d+"/"+fn0[:fn0.find("start")]+fn0[fn0.find("lambda"):]
    a=open(newname,'wb')
    cPickle.dump(newres,a)
    a.close()
