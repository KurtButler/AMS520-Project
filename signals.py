import numpy as np

def maimpute(my_data):
    for p in range(0,my_data.shape[1]):
        nanlist = np.argwhere(np.isnan(my_data[:,p]))
        if nanlist.size > 0:
            #Since I know that no NaN occur near n=0, I can take moving-averages without boundary conditions.
            for k in range(0,nanlist.size):
                my_data[nanlist[k],p] = np.mean(my_data[int(nanlist[k]-1):int(nanlist[k]),p])
    return my_data
    