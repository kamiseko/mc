# mc
mc simulation


from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import time
import timing

#Parameters
start_time=time.time()
n=10000
m=5000
k=50
theta=0.10
beta=0.03
T=1
t=0.
dt=(T-t)/float(n)
r0=0.02

#compute the real price via affine approach
def exact(k,theta,beta,T,t,r0):
    R=theta-(beta**2)/(2*(k**2))
    tau=T-t
    B=(1-np.exp(-k*tau))*(1/k)
    A=np.exp(-(R*(tau-B)+(B**2)+(beta**2)/(4*k**2)))
    return A*np.exp(-B*r0)

#mc simulation for the price
r=np.zeros((n,m),dtype=float)
r[0,:]=r0
sum1=r0
sf=np.zeros(n)
sf[0]=r0
rmean=np.zeros(n)
rmean[0]=r0
for i in xrange(1,n):
    ran=np.random.standard_normal(m)
    r[i,:]=r[i-1,:]+k*(theta-r[i-1,:])*dt + beta*(np.sqrt(dt))*ran
    sum1+=r[i,:]
    rmean[i]=mean(r[i,:])
    print i
price1=mean(np.exp(-dt*sum1))
stdd= np.std(r)
print "\n""\n",("The mc price is:%f" % price1),"\n",("The exact price is:%f"% exact(k,theta,beta,T,t,r0)),"\n",("The standard deviation is:%f"% stdd)
print "\n""\n",("--- %s seconds ---" % (time.time() - start_time))

#plot the figure
fig1=plt.figure()
a=np.arange(0,n)
b=rmean
c=r[:,0]
plt.plot(a,b,"k--")
plt.plot(a,c,"k")
plt.xlabel("time")
plt.ylabel("interest rate")
plt.title("the dynamic of interest rate under vaseick model")
fig2=plt.figure()
c=np.arange(0,m)
d=sum1
plt.plot(c,d,"k")
plt.show()
print timing
