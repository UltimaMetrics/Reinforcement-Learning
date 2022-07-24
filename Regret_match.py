import numpy as np

def regret_match(y, Pay):
    # y is a sequence of actions chosen by the 'column' player
    # Pay is a mxn matrix so that Pay(x,y) is the payout to tbe 'row'
    # player if they choose x and the column player chooses y

    # So the row player has actions {1,2,...,m} and the column
    # player has actions {1,2,...,n}

    y = y-1

    m = np.size(Pay,0)
    n = np.size(Pay,1)
    # print(m,n)
    T = len(y)
    x = np.full(T, np.nan)
    # x = x.astype(np.int32)
    u = np.full(T, np.nan)

    # start with equal choices for all row actions
    q = np.ones(m) / m
    # regret for row actions as function of time
    reg = np.zeros((T, m))

    for s in range(0,T):
        # Sample a row action from  q
        # print(q)
        x[s] = np.where(np.random.multinomial(1, q))[0]

        # row player gets paid
        # print(x[s], y[s])
        u[s] = Pay[int(x[s]), y[s]]
        # Update regret

        reg[s,:] = ( Pay[:,y[s]] - u[s] ) / (s+1)
        if s > 0:
            reg[s,:] = reg[s,:] + reg[s-1,:] * s /(s+1)
        
        # update the p.m.f. (q)
        # positive parts
        for i in range(len(reg[s,:])):
            if reg[s,i] > 0:
                q[i] = reg[s,i]
            else:
                q[i] = 0
        
        if max(q) > 0:
            # normalize to p.m.f
            q = q / sum(q)
        else:
            # If we have no regrets then play uniform i.i.d.
            q = np.ones(m) / m

       
    return reg, x, u