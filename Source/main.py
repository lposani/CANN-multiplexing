from imports import *
import simulate

t0 = time.time()

simulate.simulate(400, 0.4, 0.4, 7, 15)
### simulation results in ../Plots/Plots2D
print "time elapsed: %.3f" % (time.time() - t0)