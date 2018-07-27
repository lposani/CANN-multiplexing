from imports import *
import visualize

class network:
    def __init__(self, N):
        self.N = N
        self.J = np.zeros((N,N))
        self.h = np.zeros(N)
        self.environments = []
        self.env_dim = 1
        self.nenv = 1
        self.gamma = 0
        self.Js = []

    def random_environments(self, nenv, ndim, gamma):
        self.J = np.zeros((self.N, self.N))
        self.h = np.zeros(self.N)
        self.environments = []
        self.env_dim = ndim
        self.nenv = nenv
        self.gamma = gamma

        if ndim == 1:
            for e in range(nenv):
                self.environments.append(np.random.rand(self.N).reshape(-1,1))
                C = cdist(self.environments[e], self.environments[e])
                C = np.minimum(C, np.abs(1.-C))
                j = gamma* 20./(self.N * np.sqrt(2. * np.pi)) * np.exp( - 200 * (C**2.))
                self.Js.append(j)
                self.J += j
        if ndim==2:
            for e in range(nenv):
                self.environments.append(np.random.rand(self.N, ndim))
                C = cdist(self.environments[e], self.environments[e])
                for xpbc in [-1,0,1]:
                    for ypbc in [-1,0,1]:
                        Cpbc = cdist(self.environments[e], self.environments[e] + [xpbc, ypbc])
                        C = np.minimum(C, Cpbc)

                j = gamma* 32./(self.N * np.pi) * np.exp( - 32 * (C**2.))
                self.Js.append(j)
                self.J += j
        for i in range(self.N):
            self.J[i,i] = 0

class population:
    def __init__(self, N, mu):
        self.N = N
        self.mu = mu
        self.S = np.array(np.random.rand(N) < mu, dtype = float)
        self.network = network(self.N)
        self.saved_status = []
        self.saved_energies = []
        self.saved_energies_v = []
        self.saved_energies_m = []
        self.saved_energies_n = []

    def setup_network(self, nenv, ndim, gamma):
        self.network.random_environments(nenv, ndim, gamma)

    def energy(self, S):
        E = -0.5 * np.dot(S,np.dot(self.network.J,S))
        E -= np.dot(S, self.network.h)
        return E

    def const_metropolis_step(self, beta):
        x1 = int(np.random.rand() * self.N)
        x2 = int(np.random.rand() * self.N)
        while self.S[x1] == self.S[x2]:
            x2 = int(np.random.rand() * self.N)

        newS = np.copy(self.S)
        newS[x1] = 1 - self.S[x1]
        newS[x2] = 1 - self.S[x2]
        DE = self.energy(newS) - self.energy(self.S)

        if np.exp(- beta * DE) > np.random.rand():
            self.S[x1] = newS[x1]
            self.S[x2] = newS[x2]

    def N_const_metropolis_steps(self, beta):
        for i in range(self.N):
            self.const_metropolis_step(beta)

    def evolve(self, nstep, beta=1):
        for n in range(nstep):
            self.N_const_metropolis_steps(beta)
            self.saved_status.append(np.copy(self.S))

    def clear_memory(self):
        self.saved_status = []
        self.saved_energies = []
        self.saved_energies_v = []
        self.saved_energies_m = []
        self.saved_energies_n = []

    def randomize(self):
        self.S = np.array(np.random.rand(self.N) < self.mu, dtype=float)

class CA3:
    def __init__(self, N, ndim, nenv, gamma = 1., alpha_mec = 100, alpha_visual = 100, alpha_feedback=100, mec_sparsity=0, vis_sparsity=0, mu = 0.10):
        self.N = N
        self.mu = mu
        self.M = 'A'
        self.remap_time = []
        self.ndim = ndim
        self.nenv = nenv
        self.population = population(N, mu)
        self.population.setup_network(nenv, ndim, gamma)
        self.mec_inputs = []
        self.mec_outputs = []
        self.gamma = gamma
        self.alpha_mec = alpha_mec
        self.alpha_visual = alpha_visual
        self.alpha_feedback = alpha_feedback
        self.max_energy = 0
        self.cross_energy=0

        self.mec_sparse_vecAA = np.random.choice([0, 1], size=(self.N,), p=[mec_sparsity, 1 - mec_sparsity])
        self.mec_sparse_vecAB = np.random.choice([0, 1], size=(self.N,), p=[mec_sparsity, 1 - mec_sparsity])
        self.mec_sparse_vecBB = np.random.choice([0, 1], size=(self.N,), p=[mec_sparsity, 1 - mec_sparsity])
        self.mec_sparse_vecBA = np.random.choice([0, 1], size=(self.N,), p=[mec_sparsity, 1 - mec_sparsity])
        self.vis_sparse_vecA = np.random.choice([0, 1], size=(self.N,), p=[vis_sparsity, 1 - vis_sparsity])
        self.vis_sparse_vecB = np.random.choice([0, 1], size=(self.N,), p=[vis_sparsity, 1 - vis_sparsity])

    def mec_remapping_trial(self, gamma, x):
        IA = feedback_field(self.population.network.environments[0], x, self.alpha_feedback, self.population.S)
        IB = feedback_field(self.population.network.environments[1], x, self.alpha_feedback, self.population.S)
        DI = (IA-IB)/self.population.N
        # print DI, np.exp(-gamma*(1. + DI/2.)), np.exp(-gamma*(1. - DI/2.))

        if self.M == 'A':
            if np.random.rand() < np.exp(-gamma*(1. + DI/2.)):
                self.M = 'B'
                self.remap_time.append(len(self.population.saved_status))
                #print "MEC remapped to B!"

        elif self.M == 'B':
            if np.random.rand() < np.exp(-gamma*(1. - DI/2.)):
                self.M = 'A'
                self.remap_time.append(len(self.population.saved_status))
                #print "MEC remapped to A!"

    def mec_input(self, x0, dx, gamma):
        if self.M == 'A':
            h0 = coherent_field(self.population.network.environments[0], x0, self.alpha_mec)
            s0 = self.mec_sparse_vecAA
        if self.M == 'B':
            h0 = coherent_field(self.population.network.environments[1], x0, self.alpha_mec)
            s0 = self.mec_sparse_vecBB

        h = gamma * (h0 * s0) * self.mu
        #print "mEC energy: %.2f" % (-1*np.dot(h, self.population.S))
        self.population.saved_energies_m.append(-1. * np.dot(h, self.population.S))
        return h

    def visual_input(self, environment, x0, gamma):
        e = self.population.network.environments[environment]
        he = coherent_field(e, x0, self.alpha_visual)
        if environment==0:
            sparsevector = self.vis_sparse_vecA
        elif environment==1:
            sparsevector = self.vis_sparse_vecB
        h = gamma * sparsevector * he * self.mu
        #print "visual energy: %.2f" % (-1.*np.dot(h, self.population.S))
        self.population.saved_energies_v.append(-1.*np.dot(h, self.population.S))
        return h

    def evolve_with_mec(self, positions, dx, step_per_deltat, intensity):
        for x in positions:
            self.population.network.h = self.mec_input(x, dx, intensity)
            self.population.evolve(step_per_deltat)

    def evolve_with_visual(self, environments, positions, step_per_deltat, intensity):
        for t in range(len(positions)):
            self.population.network.h = self.visual_input(environments[t], positions[t], intensity)
            self.population.evolve(step_per_deltat)

    def evolve(self, environments, positions, mec_gamma, vis_gamma, feedback_gamma, beta, ndt =1):
        self.max_energy = max_energy(self.mu, vis_gamma, mec_gamma, self.N)
        self.cross_energy = -1*self.mu*self.N*(((mec_gamma-vis_gamma)**2)/8 + ((1-self.mu)**2)/2)
        #print self.cross_energy
        for t in range(len(positions)):
            e = environments[t]
            x = positions[t]
            h_mec = self.mec_input(x, 0.3, mec_gamma)
            h_vis = self.visual_input(e, x, vis_gamma)

            self.population.network.h = h_mec + h_vis
            self.population.evolve(ndt, beta)

            # print "network energy: %.2f" % self.network_energy()
            #
            # print "total energy: %.2f" % self.energy()
            # print "---------------------------------------------"
            #print np.dot(self.population.network.J,self.population.S)

            self.population.saved_energies_n.append(self.network_energy())
            self.population.saved_energies.append(self.energy())
            self.mec_remapping_trial(feedback_gamma, x)

    def visualize(self, title=None, positions=[], teleportation=None):
        visualize.visualize(self, title, positions, teleportation)

    def energy(self):
        E = -0.5 * np.dot(self.population.S, np.dot(self.population.network.J, self.population.S)) - np.dot(self.population.network.h, self.population.S)
        return E

    def network_energy(self):
        return -0.5 * np.dot(self.population.S, np.dot(self.population.network.J, self.population.S))

    def clear(self):
        self.population.clear_memory()
        self.population.randomize()
        self.remap_time = []

    def save_txt(self, filename):
        np.savetxt(filename+"_state.dat", self.population.saved_status)
        np.savetxt(filename+"_environments.dat", self.population.network.environments)
        np.savetxt(filename+"_Ja.dat", self.population.network.Js[0])
        np.savetxt(filename + "_Jb.dat", self.population.network.Js[1])

    def save(self, filename):
        pickle.dump(self, open("data/"+filename, 'wb'))


### Functions


def coherent_field(environment, x, alpha):
    ndim = environment[0].size
    x = np.asarray([x]).reshape(-1,ndim)
    hs = cdist(environment, x)
    if ndim==1:
        hs = np.minimum(hs, np.abs(1. - hs))
    if ndim==2:
        for xpbc in [-1, 0, 1]:
            for ypbc in [-1, 0, 1]:
                hpbc = cdist(environment, x + [xpbc, ypbc])
                hs = np.minimum(hs, hpbc)
    hs = hs.reshape(1,len(environment))[0]
    h = np.exp(-alpha * (hs **2))
    if ndim == 1:
        h *= np.sqrt(alpha/np.pi)
    if ndim == 2:
        h *= alpha/np.pi
    return h

def feedback_field(environment, x, alpha, activity):
    h=coherent_field(environment,x,alpha)
    #print np.dot(activity, h)
    return np.dot(activity, h)

def load_CA3(filename):
    C = pickle.load(open(filename))
    return C

def max_energy(f, gv, gm, N):
    r = f*N
    f = (1+f)/2
    dg = gm - gv
    e = f*f + f*(gv + gm) - dg*dg/4.
    return -r*0.466*e