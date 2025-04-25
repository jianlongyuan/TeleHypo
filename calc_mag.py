
def local_magnitude(sr, stz, stlas, stlos, xmlfiles_v, xmlfiles_h1, xmlfiles_h2, catalog, events, sonsets, v, h1, h2, points):

    delta=1/sr

    import numpy as np
    import distance
    from woodanderson import wa

    vwood=[]
    h1wood=[]
    h2wood=[]

    for inv, my_tr in zip(xmlfiles_v, v):
        try:
            my_tr.remove_response(inventory=inv, output="VEL")
            my_tr.filter('bandpass', freqmin=0.2, freqmax=20, zerophase=True)
            my_tr.detrend(type='demean')
            tmp = wa(my_tr, delta)
            vwood.append(tmp.data)

        except (KeyboardInterrupt, SystemExit):
            raise

        except:
            print('fail')
            vwood.append(np.ones(points))

    for inv, my_tr in zip(xmlfiles_h1, h1):
        try:
            my_tr.remove_response(inventory=inv, output="VEL")
            my_tr.filter('bandpass', freqmin=0.2, freqmax=20, zerophase=True)
            my_tr.detrend(type='demean')
            tmp = wa(my_tr, delta)
            h1wood.append(tmp.data)

        except (KeyboardInterrupt, SystemExit):
            raise

        except:
            print('fail')
            h1wood.append(np.ones(points))

    for inv, my_tr in zip(xmlfiles_h2, h2):
        try:
            my_tr.remove_response(inventory=inv, output="VEL")
            my_tr.filter('bandpass', freqmin=0.2, freqmax=20, zerophase=True)
            my_tr.detrend(type='demean')
            tmp = wa(my_tr, delta)
            h2wood.append(tmp.data)

        except (KeyboardInterrupt, SystemExit):
            raise

        except:
            print('fail')
            h2wood.append(np.ones(points))


    eventnum = -1
    for i in range(0, len(catalog)):
        '''
        Ml = []
        eventnum = eventnum + 1
        eventtime = catalog[i][0]
        evla = catalog[i][1]
        evlo = catalog[i][2]
        evdep = catalog[i][3]
        while events[eventnum][0] < 0:
            eventnum = eventnum + 1

        s = sonsets[eventnum]
        stnum = -1
        for j in s:
            stnum = stnum + 1
            if j < 0:
                continue
            else:
                try:
                    real_am = (h1wood[stnum][int((j - 5) * sr):int((j + 10) * sr)] ** 2 + h2wood[stnum][int((j - 5) * sr):int((j + 10) * sr)] ** 2 + vwood[stnum][int((j - 5) * sr):int((j + 10) * sr)] ** 2) ** 0.5
                except ValueError:
                    continue
                epi_dis = distance.dis(evla, evlo, stlas[stnum], stlos[stnum]) / 180 * 6371 * np.pi
                hypo_dis = (epi_dis ** 2 + ((evdep + stz[stnum]) / 1000) ** 2) ** 0.5
                R = hypo_dis
                #logA0R = 0.29 - 1.27 * 10 ** (-3) * R - 1.49 * np.log10(R)
                logA0R = 1.110 * np.log10(R/100) + 0.00189*(R-100) + 3.0          # L. K. Hutton and David M. Boore, BSSA, 1987: The Ml for southern California

                try:
                    A = max(abs(real_am))
                except ValueError:
                    continue
                Ml_i = np.log10(A) + logA0R
                Ml.append(Ml_i)

        Ml_final = np.median(Ml)
        print('Ml = ' + str(Ml_final))
        catalog[i].append(Ml_final)
        '''
        catalog[i].append(5)
    return catalog
