import numpy as np
from Markov_chain import markov_chain
from Regret_match import regret_match
import matplotlib.pyplot as plt
import statistics
from statistics import mean
# d is rock paper scissors lizard spock
# d must be an odd number
d = 21

# Generate the Payout matrix
PI = np.eye(d)
PI_item = PI[:, 1:d]
PI = np.column_stack((PI_item,PI[:,0]))
Pay = np.eye(d) - 2 * np.linalg.inv(np.eye(d) + PI)
print(Pay)

# A random stochastic matrix for the Markov chain is generated in A
A = -np.log(np.random.rand(d,d))
A = np.dot(A, np.diag(1./sum(A)))
print(A)

# number of realizations
nreal =200
# length of realizations
N = 150
# initial states of realizations
y0 = 1 + np.random.randint(d, size=(nreal,1))
# generate realizations of column player actions
y = markov_chain(N, A, y0)

# print(y)
# print(Pay)
# print(type(Pay))
# print(np.size(Pay,0))
# print(np.size(Pay,1))
y = y.astype(np.int32)

#Compute regret matching payouts
u1 = np.full((N, nreal), np.nan)
for k in range(0,nreal):
    reg, x, v = regret_match( y[:,k], Pay )
    u1[:,k] = v
    # print("k=",k)

#Q4: Plot of where column player actions are independent Markov Chain sequences
plt.figure(1)
plt.plot(np.cumsum(u1,axis=0), color = 'cornflowerblue', linewidth=1.0, linestyle="-")
plt.xlim(0, N)
# plt.ylim(-30, 30)
plt.plot(np.cumsum(np.mean(u1, axis=1)), color = 'deeppink')
plt.title('Payout sequence of independent Markov Chain')
plt.xlabel("N")
plt.ylabel("Payout")
# plt.show()

#Figure of zero average regret for u1
u1_payout = np.cumsum(u1,axis=0)
payout_hang = np.size(u1_payout,0) * 7
t = np.array(list(range(1,payout_hang + 1,7)))
t = t.reshape(-1,1)
# print(t)
plt.figure(2)
plt.plot(t, u1_payout / t, color = 'lightseagreen')
plt.xlabel("Time")
plt.ylabel("Realized regret")
plt.title('Realized regret as a function of time for many independent Markov Chain')
plt.xlim(0, 1000)



#Cumulated payout of Q4
plt.figure(3)
x = np.sum(u1,axis=0)
x = x.astype(np.int32)
xx = set(x)
plt.hist(x, bins = len(xx), color = "dodgerblue", edgecolor = 'lightpink', density = np.true_divide)
plt.xlabel("Payout")
plt.ylabel("Density")
mean_u_many_ind=mean(sum(u1))
Average_payout_ind_Markov=mean_u_many_ind
print(mean_u_many_ind)
plt.title('Cumulated payout for independent Markov Chain, Average_payout_ind_Markov = %f' %(Average_payout_ind_Markov))

#Q5: Compute regret mathcing payouts for only one realization
u2 = np.full((N, nreal), np.nan)
for k in range(0,nreal):
    reg, x, v = regret_match( y[:,0], Pay )
    u2[:,k] = v

#Plot of where column player actions are single sequence of the Markov Chain
plt.figure(4)
plt.plot(np.cumsum(u2,axis=0), color = 'cornflowerblue', linewidth=1.0, linestyle="-")
plt.xlim(0, N)
# plt.ylim(-30, 30)
plt.plot(np.cumsum(np.mean(u2, axis=1)), color = 'deeppink')
plt.title('Payout sequence of single Markov Chain')
plt.xlabel("N")
plt.ylabel("Payout")

#Figure of zero average regret for u2
u2_payout = np.cumsum(u2,axis=0)
payout_hang = np.size(u2_payout,0) * 7
t = np.array(list(range(1,payout_hang + 1, 7)))
t = t.reshape(-1,1)
plt.figure(5)
plt.plot(t, u2_payout / t, color = 'palegreen')
plt.xlim(0, 1000)
plt.xlabel("Time")
plt.ylabel("Realized regret")
plt.title('Realized regret as a function of time for single Markov Chain')


#Cumulated payout of Q5
plt.figure(6)
x = np.sum(u2,axis=0)
x = x.astype(np.int32)
xx = set(x)
plt.hist(x, bins = len(xx), color = "dodgerblue", edgecolor = 'lightpink', density = np.true_divide)
plt.xlabel("Payout")
plt.ylabel("Density")
mean_u_one=mean(sum(u2))
Average_payout_one_Markov=mean_u_one
print(mean_u_one)
plt.title('Cumulated payout for single Markov Chain, Average_payout_one_Markov = %f' %(Average_payout_one_Markov))


plt.show()
