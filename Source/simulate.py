from imports import *
import network
import visualize
from utilities import *
t0 = time.time()

# taking values from real CA3 data
steps_per_theta = 4
teletimes = steps_per_theta * np.loadtxt('../Data/tele_times_theta.txt', dtype='int')
x = np.loadtxt('../Data/positions_x_%u.txt' % steps_per_theta)
y = np.loadtxt('../Data/positions_y_%u.txt' % steps_per_theta)
positions = np.transpose(np.asarray([x,y]))
T = len(positions)
light_conditions = np.zeros(T)
for i in range(7):
    light_conditions[teletimes[2*i]:teletimes[2*i+1]] = 1
light_conditions = np.asarray(light_conditions, dtype='int')
v = np.sqrt(np.sum((positions[1:]-positions[:-1])**2,1))
vmeans = []
for t in teletimes[:-1]:
    vmeans.append(np.mean(v[t:t+600]))

class simulation(threading.Thread):
    def __init__(self, N, mec_gamma, vis_gamma, feedback_gamma, beta, mec_s=0, vis_s=0):
        threading.Thread.__init__(self)
        self.N = N
        self.mec_gamma = mec_gamma
        self.vis_gamma = vis_gamma
        self.feedback_gamma = feedback_gamma
        self.beta = beta
        self.ms = mec_s
        self.vs = vis_s

    def run(self):
        simulate(self.N, self.mec_gamma, self.vis_gamma, self.feedback_gamma, self.beta, self.ms, self.vs, True, '../Plots/Videos2')

def simulate(N, mec_gamma, vis_gamma, feedback_gamma, beta, mec_s=0, vis_s=0, plotall=False, folder='../Plots/Plots2D'):
    C = network.CA3(N, 2, 2, alpha_visual=32, alpha_mec=32, alpha_feedback=32, mec_sparsity=mec_s, vis_sparsity=vis_s)
    name = "g_%f_b_%u_mg_%.2f_vg_%.2f_fg_%.1f" % (vis_gamma/mec_gamma, beta, mec_gamma, vis_gamma, feedback_gamma)
    mkdir(folder + '/' + name)
    mkdir(folder + '/' + name + '/data')

    for i in range(14):
        print "starting simulations %u for parameters " % i + name
        C.clear()
        if i%2 ==1:
            C.M='B'
        elif i%2 ==0:
            C.M='A'
        tstart = teletimes[i]-200
        tend = teletimes[i]+800
        #print tstart
        #print tend
        C.evolve(light_conditions[tstart:tend],positions[tstart:tend], mec_gamma=mec_gamma, vis_gamma=vis_gamma, feedback_gamma=feedback_gamma, beta=beta)
        print "printing plots for %u neurons in folder " % N + name
        visualize.visualize_2D(C, folder + '/' + name + '/%u_%u' % (N,i), positions[tstart:tend], [200], plotall=plotall)
        # pickle.dump(C, open('Plots2D/'+name+'/data/%u_%u' % (N,i), 'w'))
        # np.savetxt('Plots2D/'+name+'/data/%u_%u' % (N,i), C.population.saved_status, '%u')
        # np.savetxt('Plots2D/' + name + '/data/%u_%u_rt' % (N, i), C.remap_time, '%u')
        #return C

def simulate_many(Ns, mgs, vgs, fgs, bs, mec_s=0, vis_s=0):
    threads = []
    for N in Ns:
        for mg in mgs:
            for vg in vgs:
                for fg in fgs:
                    # choosing parameters
                    for b in bs:
                        threads.append(simulation(N, mg, vg, fg, b, mec_s, vis_s))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

def simulate_many_fixg(Ns, g, mgs, fgs, bs, mec_s=0, vis_s=0):
    threads = []
    vgs = g * np.asarray(mgs)
    for N in Ns:
        for i in range(len(mgs)):
            for fg in fgs:
                    # choosing parameters
                for b in bs:
                    threads.append(simulation(N, mgs[i], vgs[i], fg, b, mec_s, vis_s))
    for t in threads:
        t.start()
    for t in threads:
        t.join()