from imports import *
from utilities import *
import network
import matplotlib.animation as animation

def visualize(CA3, title, positions = [], teleportations=[]):
    if CA3.population.network.env_dim == 1:
        visualize_1D(CA3, title, positions, teleportations)
    if CA3.population.network.env_dim == 2:
        visualize_2D(CA3, title, positions, teleportations)

def visualize_1D(CA3, title, gamma_m, gamma_v, positions=[], teleportations=[]):
    S = CA3.population
    remap = CA3.remap_time
    n = S.N
    T = len(S.saved_status)
    E = np.zeros((S.network.nenv, T))
    dx = 1.
    P = np.asarray(S.saved_status)
    d = np.transpose(S.saved_status)
    plt.figure(figsize=(15, 5 * S.network.nenv))
    for environment in range(S.network.nenv):
        ax = plt.subplot(S.network.nenv+1, 1, environment+1)
        e = S.network.environments[environment]
        plt.scatter(repmat(range(T), 1, n), zero_to_nan(e * d), marker='x', c='k', s=50./n)
        if len(positions):
            plt.plot(range(T), positions, 'C1--')
        for t in range(T):
            E[environment, t] = - 0.5 * np.dot(P[t],np.dot(S.network.Js[environment],P[t]))
        if environment==0:
            plt.title(r"environment #%u      $\gamma_V$ = %.2f    $\gamma_M$ = %.2f     N = %u " % (environment, gamma_v, gamma_m, n))
        else:
            plt.title("environment #%u                                            " % environment)

        if len(teleportations):
            for teleportation in teleportations:
                plt.plot([teleportation, teleportation], [0,1], 'r')
        if len(remap):
            for r in remap:
                plt.plot([r, r], [0,1], 'g')
        plt.yticks([])
        plt.xticks([])
        plt.ylabel("place-field positions")
        #plt.xlabel("MC time")
        plt.xlim([-1, T+1])
        plt.ylim([0,1])
    ax = plt.subplot(S.network.nenv + 1, 1, S.network.nenv + 1)

    for e in range(S.network.nenv):
        plt.plot(range(T), E[e], label = "environment #%u"%e)
    plt.ylabel('Projection energy')
    plt.legend()
    plt.xlim([-1, T+1])
    plt.xlabel("MC time")
    plt.yticks([])
    y = plt.ylim()

    if len(teleportations):
        for teleportation in teleportations:
            plt.plot([teleportation, teleportation], y, 'r')
    if len(remap):
        for r in remap:
            plt.plot([r, r], y, 'g')

    plt.ylim(y)
    if title:
        #plt.savefig("Plots1D/"+title+".pdf")
        plt.savefig("../Plots/Plots1D/" + title + ".png")
    else:
        plt.savefig("../Plots/Plots1D/evolve.pdf")
    plt.close()

    nenv = S.network.nenv

    E = np.zeros((S.network.nenv, T))
    for t in range(T):
        for env in range(nenv):
            E[env, t] = - 0.5 * np.dot(S.saved_status[t], np.dot(S.network.Js[env], S.saved_status[t]))

    f, (a, a1, a2) = plt.subplots(3, sharex=True)
    f.set_size_inches(10, 10)
    for e in range(S.network.nenv):
        a.plot(range(T), E[e], label="environment #%u" % e)
    a.set_ylabel('Projection energy')
    a.legend()
    a.set_xlim([-1, T + 1])
    # a.plot([-1,T], [CA3.cross_energy, CA3.cross_energy], '--k')

    # a.set_xlabel("MC time")
    # a.set_yticks([])
    y = a.get_ylim()
    if len(teleportations):
        for teleportation in teleportations:
            a.plot([teleportation, teleportation], y, 'r')
    if len(remap):
        for r in remap:
            a.plot([r, r], y, 'g')
    a.set_ylim(y)
    a1.plot(S.saved_energies, 'k', label='total')
    a1.plot([-1, T], [CA3.max_energy, CA3.max_energy], '--k')
    a1.set_ylabel('Energy')

    a2.plot(S.saved_energies_v, 'b', label='visual')
    a2.plot(S.saved_energies_m, 'g', label='mec')
    a2.plot(S.saved_energies_n, 'r', label='network')
    a2.legend()
    a2.set_ylabel('Energy')
    if title:
        f.savefig("../Plots/Plots1D/" + title + "_energy.pdf")
    else:
        f.savefig("../Plots/Plots1D/energy.pdf")
    plt.close(f)

def visualize_2D(CA3, title, positions=[], teleportations=[], plotall= False, fps=32):
    envname=['A','B','C','D']
    S = CA3.population
    T = len(S.saved_status)
    print T
    nenv = S.network.nenv
    remap = CA3.remap_time

    E = np.zeros((S.network.nenv, T))
    for t in range(T):
        for env in range(nenv):
            E[env, t] = - 0.5 * np.dot(S.saved_status[t], np.dot(S.network.Js[env], S.saved_status[t]))

    f,(a, a0, a1, a2) = plt.subplots(4, sharex=True)
    f.set_size_inches(10,10)
    for e in range(S.network.nenv):
        a.plot(range(T), E[e], label="environment #%u" % e)
    a.set_ylabel('Projection energy')
    a.legend()
    a.set_xlim([-1, T + 1])
    # a.plot([-1,T], [CA3.cross_energy, CA3.cross_energy], '--k')

    #a.set_xlabel("MC time")
    #a.set_yticks([])
    y = a.get_ylim()
    if len(teleportations):
        for teleportation in teleportations:
            a.plot([teleportation, teleportation], y, 'r')
    if len(remap):
        for r in remap:
            a.plot([r, r], y, 'g')
    a.set_ylim(y)

    DE = DL(CA3)
    a0.bar(np.arange(len(DE)), DE)
    a0.set_ylabel('$\Delta L$')

    a1.plot(S.saved_energies, 'k', label='total')
    a1.plot([-1,T], [CA3.max_energy, CA3.max_energy], '--k')
    a1.set_ylabel('Energy')

    a2.plot(S.saved_energies_v, 'b', label='visual')
    a2.plot(S.saved_energies_m, 'g', label='mec')
    a2.plot(S.saved_energies_n, 'r', label='network')
    a2.legend()
    a2.set_ylabel('Energy')
    if title:
        f.savefig(title + "_energy.pdf")
    else:
        f.savefig("../Plots/Plots2D/energy.pdf")
    plt.close(f)

    # nb = raw_input('Should I print all the frames?\n')
    # if nb=='n' or nb=='no':
    #     plotall=0

    if plotall:
        ff = plt.figure(figsize=(5 * nenv, 7.5))
        # mkdir('Plots2D/'+title+'_all')
        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=fps)

        with writer.saving(ff, title+'.mp4', dpi=100):
            cmap = plt.get_cmap('jet')
            x = np.transpose(S.network.environments[0])[0]
            y = np.transpose(S.network.environments[0])[1]
            y1 = np.transpose(S.network.environments[1])[1]
            c = np.asarray(cmap(np.sin(np.pi*x)*np.sin(np.pi*y)))
            # c = np.transpose([(x + y) / 2, 0.1 + np.zeros(len(x)), (2 - (x + y)) / 2])
            # c = np.transpose([x, y1, 0.5*np.ones(len(x))])
            for t in range(T):
                print t
                for env in range(nenv):
                    # ax = plt.subplot2grid((3, 2*nenv), (0,2*env), colspan=2, rowspan=2)
                    subplotspec = plt.GridSpec(3, 2*nenv).new_subplotspec((0,2*env), rowspan=2,colspan=2)
                    ax = ff.add_subplot(subplotspec)
                    ax.set_title("environment " + envname[env])
                    e = S.network.environments[env]
                    ez = zero_to_nan(S.saved_status[t] * np.transpose(e))
                    ax.scatter(ez[0], ez[1], alpha=0.5, edgecolors='none', s=700)
                    if len(positions):
                        ax.plot(positions[t, 0], positions[t, 1], 'x', color='r', markersize=10)
                    ax.set_xlim([0,1])
                    ax.set_ylim([0,1])

                #ax = plt.subplot2grid((3, 2*nenv), (2, 0), colspan=2*nenv)
                subplotspec = plt.GridSpec(3, 2*nenv).new_subplotspec((2, 0), rowspan=1, colspan=2*nenv)
                ax = ff.add_subplot(subplotspec)
                for e in range(S.network.nenv):
                    ax.plot(range(T), E[e], label="environment " + envname[e])
                ax.set_ylabel('Projection energy')
                ax.legend()
                ax.set_xlim([-1, T + 1])
                ax.set_xlabel("MC time")
                ax.set_yticks([])
                y = ax.get_ylim()
                if len(teleportations):
                    for teleportation in teleportations:
                        ax.plot([teleportation, teleportation], y, 'r')
                if len(remap):
                    for r in remap:
                        ax.plot([r, r], y, 'g')
                ax.set_ylim(y)
                ax.plot([t, t], y, 'k', linewidth=2)
                #plt.show()
                #plt.savefig('Plots2D/'+title+'_all/%u.pdf' % t)
                #plt.close()
                writer.grab_frame()
                ff.clear()
        plt.close(ff)

def load_and_plot(filename, title, positions, teleportations):
    C = pickle.load(open(filename))
    visualize_2D(C, title, positions, teleportations)