from analysis import *
import simulate
from utilities import *
from fit import *
import network


def correlation_diagram(gvs, betas, N):
    corr_diagram = np.zeros((len(gvs), len(betas)))
    pos_stab_diagram = np.zeros((len(gvs), len(betas)))
    pos_unstab_diagram = np.zeros((len(gvs), len(betas)))
    dl_magnitude = np.zeros((len(gvs), len(betas)))
    n_trials = 15
    for i in range(len(gvs)):
        for j in range(len(betas)):

            DT = 500
            C = network.CA3(N, 2, 2, alpha_visual=32, alpha_mec=32, alpha_feedback=32, mec_sparsity=0, vis_sparsity=0)
            f = np.zeros(n_trials*DT)
            dls = np.zeros(n_trials*DT)
            fstab = np.zeros(n_trials*100)
            er_is_unst=[]
            er_is_stab=[]
            print '--------------------- gamma %.2f beta %.u' % (gvs[i], betas[j])
            for k in range(n_trials):
                C.clear()
                t = teletimes[k]

                print 'analysis of teleportation %u' % k

                if j % 2 == 1:
                    C.M = 'B'
                elif j % 2 == 0:
                    C.M = 'A'
                pos = positions[t - 100:t + DT]
                light = light_conditions[t - 100:t + DT]
                C.evolve(light, pos, mec_gamma=gvs[i], vis_gamma=gvs[i], feedback_gamma=1000, beta=betas[j])
                DE = DL(C)
                DL_plot(DE, 4, 'Diagram/deltaL/%.3f_%.3f_%u' % (gvs[i], betas[j], k), 100, remap=[])
                dls[k*DT : (k+1)*DT] = DE[100:]

                if j % 2 == 1:
                    f[k*DT : (k+1)*DT] = DE[100:] < 0
                    fstab[k * 100: (k + 1) * 100] = DE[:100] < 0
                elif j % 2 == 0:
                    f[k*DT : (k+1)*DT] = DE[100:] > 0
                    fstab[k * 100: (k + 1) * 100] = DE[:100] > 0

                state = C.population.saved_status
                posA, posB = infer_positions(state, C.population.network.environments[0], C.population.network.environments[1])
                IS = DE < 0  # 0 for A and 1 for B
                pos_IS = np.where(np.transpose(repmat(IS, 2, 1)), posB, posA)
                err_IS = pbc_distance(pos, pos_IS)
                er_is_unst.append(err_IS[100:])
                er_is_stab.append(err_IS[:100])


            er_is_unst = np.reshape(er_is_unst, n_trials * DT)
            er_is_stab = np.reshape(er_is_stab, n_trials * 100)

            #c = time_correlation(f, 60) / 4.
            [sa, sb] = compute_soujourn_times(dls)

            print 0.5*(np.mean(sa)+np.mean(sb))/4., 1./np.mean(er_is_stab), 1./np.mean(er_is_unst), np.mean(np.abs(dls))

            corr_diagram[i,j] = 0.5*(np.mean(sa)+np.mean(sb))/4.
            pos_stab_diagram[i,j] = 1./np.mean(er_is_stab)
            pos_unstab_diagram[i,j] = 1./np.mean(er_is_unst)
            dl_magnitude[i,j] = np.mean(np.abs(dls))

            #plt.figure()
            #plt.plot(np.linspace(1,len(c), len(c)),c, 'o')
            #plt.savefig('../Plots/Diagram/time_correlations/%.2f_%.2f.pdf' % (gvs[i], betas[j]))
            #plt.close()
    np.savetxt('../Data/Diagram/sojourn.txt', corr_diagram)
    np.savetxt('../Data/Diagram/gs.txt', gvs)
    np.savetxt('../Data/Diagram/betas.txt', betas)
    np.savetxt('../Data/Diagram/pos_stab_diagram.txt', pos_stab_diagram)
    np.savetxt('../Data/Diagram/pos_unst_diagram.txt', pos_unstab_diagram)
    np.savetxt('../Data/Diagram/abs_dl.txt', dl_magnitude)

    plt.figure(figsize=(8, 6))
    plt.pcolor(np.transpose(corr_diagram), norm=LogNorm(vmin=np.min(np.min(corr_diagram)), vmax=np.max(np.max(corr_diagram))))
    plt.title('mean sojourn time diagram')
    plt.ylabel(r'$\beta$')
    plt.xlabel('$\gamma_V / \gamma_J$')
    plt.xticks(np.linspace(0,(len(gvs))-1,(len(gvs)))+0.5, gvs)
    plt.yticks(np.linspace(0,(len(betas))-1,(len(betas)))+0.5, betas)
    plt.colorbar()

    plt.figure(figsize=(8, 6))
    plt.pcolor(np.transpose(pos_stab_diagram))
    plt.title('navigation stability (1 / mean error) stable conditions')
    plt.ylabel(r'$\beta$')
    plt.xlabel('$\gamma_V / \gamma_J$')
    plt.xticks(np.linspace(0, (len(gvs)) - 1, (len(gvs))) + 0.5, gvs)
    plt.yticks(np.linspace(0, (len(betas)) - 1, (len(betas))) + 0.5, betas)
    plt.colorbar()

    plt.figure(figsize=(8, 6))
    plt.pcolor(np.transpose(pos_unstab_diagram))
    plt.title('navigation stability (1 / mean error) conflict conditions')
    plt.ylabel(r'$\beta$')
    plt.xlabel('$\gamma_V / \gamma_J$')
    plt.xticks(np.linspace(0, (len(gvs)) - 1, (len(gvs))) + 0.5, gvs)
    plt.yticks(np.linspace(0, (len(betas)) - 1, (len(betas))) + 0.5, betas)
    plt.colorbar()

    plt.figure(figsize=(8, 6))
    plt.pcolor(np.transpose(dl_magnitude))
    plt.title('bump completeness (|$\Delta$|L) diagram')
    plt.ylabel(r'$\beta$')
    plt.xlabel('$\gamma_V / \gamma_J$')
    plt.xticks(np.linspace(0, (len(gvs)) - 1, (len(gvs))) + 0.5, gvs)
    plt.yticks(np.linspace(0, (len(betas)) - 1, (len(betas))) + 0.5, betas)
    plt.colorbar()

    plt.show()