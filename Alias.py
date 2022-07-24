import math
import numpy as np
from numpy.core.records import array

def alias(p, *argu):
    # creates a sampler for the categorical distribution p
    # repeats the probabilities to create a faster sampler
    # but does not affect the resulting distribution
    n = len(p)

    # Query whether to feed the number of repeats, if not, the default number is automatically generated according to n
    if len(argu) == 0:
        repeats = min(100, math.ceil(2e6/n))
        # cache size guess about 16MB
    else:
        repeats = argu
    
    # Given a PMF
    p = np.array(p)
    p = p.reshape(-1,1)

    # Decide which is output if failed
    a = np.zeros([n,1])
    # probabiliities of success
    q = np.ones([n,1])

    # The two intermediate variables here have no practical meaning and are only convenient for subsequent program processing
    remain = range(0,n)
    remain = np.array(remain)
    done = np.zeros([n,1])
    # construct the alias table using the 'Robin Hood' algorithm

    # The loop termination condition is that the length of the  variable is 1
    while len(remain) >= 2:
        pm = np.min(p); jm = np.where(p == pm); jm = jm[0][0]
        pM = np.max(p); jM = np.where(p == pM); jM = jM[0][0]
        a[remain[jm]] = remain[jM]
        q[remain[jm]] = n * pm
        p[jm] = 1/n
        p[jM] = pM +pm - 1/n
        remain_index = np.where(remain != remain[jm])
        remain = remain[remain_index]
        p = np.vstack((p[0:jm], p[jm:-1]))
        # Pick p when the loop ends

    # apply repeats - faster algorithm than the equivalent padding
    j = np.ceil( np.array(range(1,(n*repeats+1))) / repeats)
    j = j-1
    # Reshape function is designed to convert between row vectors and column vectors
    j = j.reshape(len(j), -1)
    # astype function is to make sure the matrix is an integer type to avoid bug
    
    j = j.astype(np.int32)
    # Get remainder
    k = np.mod( np.array(range(0,n*repeats)), repeats )
    k = k.reshape(len(k), -1)
    j = j.reshape(1,len(j))
    j = j[0]
    qq = repeats * q[j] - k

    # Find the poistion of element less than 0 in qq
    # Assign the corresponding element in a to the corresponding element in matrix j
    qq_index = np.where(qq <= 0)
    qq_index = qq_index[0]
    a = a.reshape(1,len(a))
    a = a[0]
    j[qq_index] = a[j[qq_index]]

    # Fnd all the elements within 0-1 in the qq matrix
    # If it exceeds the range of 0-1, if it is less than 0, it is 0, and if it is greater than 1, it is 1.
    # The above elements are stored in qq_for_fpq for backup
    qq_for_fpq = []
    qq = qq.reshape(1,len(qq))[0]
    for i in range(len(qq)):
        if qq[i] < 0:
            qq_for_fpq.append(0)
        elif qq[i] > 1:
            qq_for_fpq.append(1)
        else:
            qq_for_fpq.append(qq[i])

    # Use mod function
    # Take out the decimal part of the element in qq_for_fpq
    for i in range(len(qq_for_fpq)):
        qq_for_fpq[i] = np.mod( qq_for_fpq[i], 1)
    fpq = qq_for_fpq

    # the cases which have stochastic outcomes
    ju = j[np.where(np.array(fpq) != 0)]
    # how many are there generically n
    nunc = len(ju)
    # result if flip
    au = a[ju]
    # print(au)
    # flip threshold
    qu = qq[np.where(np.array(fpq) != 0)]

    # Find the element with fpq=0
    # Combine the elements corresponding to fpq=0 in ju and j
    j_for_js = []
    for i in range(len(fpq)):
        if fpq[i] == 0:
            j_for_js.append(i)
    js = np.hstack((ju, j[j_for_js]))

    # Function handle for the random number generation
    # Follow Professor's example code
    
    def tmp(x):
        w = np.random.randint(1, n*repeats + 1, size=(1,x))[0]
        y = js[w-1]
        fu = np.where(w <= nunc)
        fu = fu[0]
        fufu = []
        if len(fu) > 0:
            for i in range(len(fu)):
                if np.random.rand() > qu[w[fu[i]] - 1]:
                    fufu.append(fu[i])
        fu = np.array(fufu)
        fu = fu.astype(np.int32)
        y[fu - 1] = au[w[fu] - 1]
        # print(y)
        return y

    return tmp
    # return 
    