from imports import *
import network
import visualize
from utilities import *
from scipy import stats
t0 = time.time()

# taking values from real CA3 data

steps_per_theta = 4

light_conditions = load_light_conditions(steps_per_theta)
positions = load_positions(steps_per_theta)
teletimes = load_teletimes(steps_per_theta)

v = np.sqrt(np.sum((positions[1:]-positions[:-1])**2,1))
vmeans = []
for t in teletimes[:-1]:
    vmeans.append(np.mean(v[t:t+600]))

def panel_flickering_rate_routine(N, ntrials, gm, gv, gf, beta):
    flickering_time(N, ntrials, gm, gv, gf, beta)
    remap_times(ntrials, N, gm, gv, gf, beta)
    sojourn_times(ntrials, N, gm, gv, beta)

def flickering_time(N, ntrials, gm, gv, gf, beta, DT = 352):

    # teleportation sessions
    mkdir('../Plots/Analysis/time_analysis/%.1f_%u_%.2f_%.2f' % (gf, N, gm, gv))
    rates = []
    dt = 32.0
    t_before = 32*3
    T = DT+t_before
    alive = np.zeros(int(T / dt))
    for j in range(ntrials):
        C = network.CA3(N, 2, 2, alpha_visual=32, alpha_mec=32, alpha_feedback=32, mec_sparsity=0, vis_sparsity=0)
        for i in range(15):
            t = teletimes[i]
            #light_conditions = np.asarray(np.hstack([np.zeros(t_before), np.ones(DT)]), dtype=int)
            #positions = np.transpose(np.vstack([np.linspace(0, 1, t_before + DT), np.linspace(0, 1, t_before + DT)]))
            # t = t_before
            t0 = time.time()
            if i % 2 == 1:
                C.M = 'B'; M='A'
            elif i % 2 == 0:
                C.M = 'A'; M='B'

            C.clear()
            C.evolve(light_conditions[t - t_before:t + DT], positions[t - t_before:t + DT], mec_gamma=gm, vis_gamma=gv, feedback_gamma=gf, beta=beta)

            DE = DL(C)
            # DL_plot(DE, dt, 'time_analysis/%.1f_%u_%.2f_%.2f/t%u_%u_' % (gf, N, gm, gv, i, j), t_before, C.remap_time)
            visualize.visualize_2D(C, '../Plots/Analysis/time_analysis/%.1f_%u_%.2f_%.2f/t%u_%u_' % (gf, N, gm, gv, i, j), positions[t - t_before + DT], [t_before], plotall=False)
            rates_i, alive_i = flickering_rate_time(DE, C.remap_time, M, dt)
            alive = alive + np.asarray(alive_i)
            rates.append(rates_i)
            print "single network time: %.3f s" % (time.time() - t0)
            rates_i[0:int(t_before/dt)] = 1-rates_i[0:int(t_before/dt)]

    #plt.plot(np.linspace(-t_before/dt,DT/dt,T/dt), np.mean(rates, 0))
    plt.figure(figsize=[3,3])
    x = np.linspace(-t_before / dt, DT / dt, T / dt)
    plt.bar(x,np.mean(rates,0),color='k')
    plt.xlim([-3.5,10.5])
    plt.ylim([0,1])
    plt.xlabel('time from teleportation (s)')
    plt.ylabel('flickering rate')
    plt.title('model: not normalized')

    plt.figure(figsize=[3, 3])
    plt.bar(x, 15*ntrials*np.mean(rates, 0)/alive, color='k')
    plt.xlim([-3.5, 10.5])
    plt.ylim([0, 1])
    plt.xlabel('time from teleportation (s)')
    plt.ylabel('flickering rate')
    plt.title('model: normalized')
    np.savetxt('../Data/rates_%u_%.2f.txt' % (N, gf), rates)
    np.savetxt('../Data/alive_%u_%.2f.txt' % (N, gf), alive)
    return rates, alive

def rate_analysis(N, ntrials, gm, gv, beta, DT = 600):
    mkdir('../Plots/Analysis/rate_analysis/DL/%u_%.2f_%.2f' % (N, gm, gv))
    vmeans = []
    for t in teletimes[:-1]:
        vmeans.append(np.mean(v[t:t + DT]))
    rates = []
    permanence = []
    for j in range(15):
        t = teletimes[j]
        rate_t = []
        for i in range(ntrials):
            print 'iteration %u of teleportation %u' % (i,j)
            C = network.CA3(N, 2, 2, alpha_visual=32, alpha_mec=32, alpha_feedback=32, mec_sparsity=0, vis_sparsity=0)
            if j % 2 == 1:
                C.M = 'B'
            elif j % 2 == 0:
                C.M = 'A'
            C.evolve(light_conditions[t-100:t+DT], positions[t-100:t+DT], mec_gamma=gm, vis_gamma=gv, feedback_gamma=800, beta=beta) # extreme gamma to avoid mEC remapping
            state = C.population.saved_status[100:]
            EA = np.zeros(DT)
            EB = np.zeros(DT)
            for k in range(DT):
                EA[k] = -0.5 * np.dot(state[k], np.dot(C.population.network.Js[0], state[k]))
                EB[k] = -0.5 * np.dot(state[k], np.dot(C.population.network.Js[1], state[k]))
            DE = EA-EB
            DL_plot(DE, 32, 'rate_analysis/DL/%u_%.2f_%.2f/t%u_%u' % (N, gm, gv,j,i), 0)
            signchanges = sign_changes(DE, 15, C.M) # smooth signal and compute transitions
            pt = permanence_times(DE)
            if j % 2 == 1:
                permanence.append([pt[0], pt[1]])
            elif j % 2 == 0:
                permanence.append([pt[1], pt[0]])
            rate_t.append(signchanges)
            #print signchanges
        rates.append(rate_t)
    rmeans = np.mean(rates,1)

    plt.errorbar(vmeans, rmeans, np.std(rates,1)/np.sqrt(ntrials), fmt='o')
    plt.xlabel('mean session velocity')
    plt.ylabel('mean transition rate')
    r,p = scipy.stats.pearsonr(vmeans, rmeans)
    plt.title('R = %.2f    p = %.5f' % (r,p))
    plt.savefig('../Plots/Analysis/rate_analysis/rate-v_%.2f_%.2f.pdf' % (gm, gv))
    plt.close()
    return rmeans, permanence

def rate_analysis_constx(N, ntrials, gm, gv, beta, DT = 600):
    permanence = []
    rate_A = []
    rate_B = []
    pos = np.repeat([[0.5,0.5],[0.5,0.5]],(100+DT)/2,0)
    mkdir('../Plots/Analysis/rate_analysis/DL/%u' % N)
    for i in range(ntrials):
        print 'iteration %u' % i
        C = network.CA3(N, 2, 2, alpha_visual=32, alpha_mec=32, alpha_feedback=32, mec_sparsity=0, vis_sparsity=0)
        C.M = 'A'
        lc = light_conditions[teletimes[0] - 100 : teletimes[0] + DT]
        C.evolve(lc, pos, mec_gamma=gm, vis_gamma=gv, feedback_gamma=800, beta=beta) # extreme gamma to avoid mEC remapping
        DE = DL(C)
        DL_plot(DE, 32, 'rate_analysis/DL/%u/%.2f_%.2f_%u' % (N, gm, gv,i), 100)
        signchanges_A = sign_changes(DE[200:], 15, 'A') # smooth signal and compute transitions
        signchanges_B = sign_changes(DE[200:], 15, 'B') # smooth signal and compute transitions
        pt = permanence_times(DE)
        if i % 2 == 1:
            permanence.append([pt[0], pt[1]])
        elif i % 2 == 0:
            permanence.append([pt[1], pt[0]])
        #print signchanges
        rate_A.append(signchanges_A)
        rate_B.append(signchanges_B)
    print rate_A
    print rate_B
    return rate_A, rate_B, permanence

def permanence_analysis(N, ntrials, gms, gvs):
    theorate_A = []
    theorate_B = []

    rates_A = []
    rates_B = []
    stdr_A = []
    stdr_B = []

    deltagamma = []
    permanence = []

    for gm in gms:
        for gv in gvs:

            rate, p = rate_analysis(N, ntrials, gm, gv, 20, DT = 600)

            #theorate_A.append(np.exp(0.25*(gm-gv)**2.) * np.exp(0.45 * (gm-gv)))

            theorate_A.append(gm/gv)

            # theorate_B.append(np.exp(-0.25*(gm-gv)**2.) * np.exp(-0.45 * (gm-gv)))
            # rates_A.append(np.mean(rates_A))
            # rates_B.append(np.mean(rates_B))
            # stdr_A.append(np.std(rates_A)/np.sqrt(ntrials))
            # stdr_B.append(np.std(rates_B)/np.sqrt(ntrials))
            # print theorate_A
            # print rates_A
            # print stdr_A


            deltagamma.append(np.exp(gm-gv))
            m = np.mean(p,0)
            #permanence.append(m[0]/m[1])
            permanence.append(m[0])

    plt.figure()
    plt.plot(theorate_A, permanence, '-o')
    plt.xlabel('gm/gv')
    plt.ylabel('flickering rate')

    #r, p = scipy.stats.pearsonr(deltagamma, permanence)
    #plt.title('R = %.2f    p = %.5f' % (r, p))

    plt.savefig('../Plots/Analysis/rate_analysis/permanence.pdf')
    np.savetxt('../Data/permanence_flickering.txt', permanence)
    np.savetxt('../Data/permanence_gm/gv.txt', theorate_A)

    # plt.figure()
    # plt.errorbar(rates_A, theorate_A, stdr_A, '-o')
    # plt.errorbar(rates_B, theorate_B, stdr_B, '-o')
    # plt.xlabel('transition rate (theoretical)')
    # plt.ylabel('transition rate (model)')
    # # r, p = scipy.stats.pearsonr(coshs, frequency)
    # # plt.title('R = %.2f    p = %.5f' % (r, p))
    # plt.savefig('../Plots/Analysis/rate_analysis/frequency.pdf')
    return (deltagamma, permanence, rates_A, rates_B, theorate_A, theorate_B)

def position_analysis(N, gm, gv, beta, DT = 600, idx=0, gf=1000):
    path = '../Plots/Analysis/position_analysis/N%u_M%.2f_V%.2f' % (N, gm, gv)
    mkdir(path)
    er_is_unst = []
    er_is_stab = []
    er_nis_unst = []
    er_nis_stab = []
    er_na_unst = []
    er_na_stab = []

    C = network.CA3(N, 2, 2, alpha_visual=32, alpha_mec=32, alpha_feedback=32, mec_sparsity=0, vis_sparsity=0)
    for j in range(15):
        t = teletimes[j]
        print 'analysis of teleportation %u' % (j)
        C = network.CA3(N, 2, 2, alpha_visual=32, alpha_mec=32, alpha_feedback=32, mec_sparsity=0, vis_sparsity=0)
        if j % 2 == 1:
            C.M = 'B'; M='B'
        elif j % 2 == 0:
            C.M = 'A'; M='A'
        pos = positions[t - DT:t + DT]
        light = light_conditions[t - DT:t + DT]
        C.evolve(light, pos, mec_gamma=gm, vis_gamma=gv, feedback_gamma=gf, beta=beta)
        state = C.population.saved_status

        posA, posB = infer_positions(state, C.population.network.environments[0], C.population.network.environments[1])
        DE = DL(C)

        IS = DE < 0 # 0 for A and 1 for B
        pos_IS = np.where(np.transpose(repmat(IS,2,1)), posB, posA)
        pos_nIS = np.where(np.transpose(repmat(IS,2,1)), posA, posB)
        if M=='A':
            pos_NA = np.vstack([posA[:DT,:], posB[DT:, :]])
        if M=='B':
            pos_NA = np.vstack([posB[:DT,:], posA[DT:, :]])

        err_IS = pbc_distance(pos, pos_IS)
        err_nIS = pbc_distance(pos, pos_nIS)
        err_NA = pbc_distance(pos, pos_NA)

        plot_position(pos, pos_IS, path + '/%u.pdf' % j)
        visualize.visualize_2D(C, path + '/%u' % j, pos, [DT], plotall=False)

        er_is_unst.append(err_IS[DT:])
        er_is_stab.append(err_IS[:DT])
        er_nis_unst.append(err_nIS[DT:])
        er_nis_stab.append(err_nIS[:DT])
        er_na_unst.append(err_NA[DT:])
        er_na_stab.append(err_NA[:DT])

    er_is_unst = np.reshape(er_is_unst, 15*DT)
    er_is_stab = np.reshape(er_is_stab, 15*DT)
    er_nis_unst = np.reshape(er_nis_unst, 15*DT)
    er_nis_stab = np.reshape(er_nis_stab, 15*DT)
    er_na_unst = np.reshape(er_na_unst, 15*DT)
    er_na_stab = np.reshape(er_is_stab, 15*DT)

    # gaussian kernel estimate
    x = np.linspace(0,1,1000)

    plt.plot(x, stats.gaussian_kde(er_is_unst).evaluate(x), label='Internal map - Unstable', linewidth=2)
    plt.plot(x, stats.gaussian_kde(er_is_stab).evaluate(x), label='Internal map - Stable', linewidth=2)
    plt.legend()
    plt.xlim([0,1])
    plt.xlabel('position error')
    plt.savefig(path + '_gkde_IS_%u.pdf' % idx)
    plt.close()

    plt.figure()
    plt.plot(x, stats.gaussian_kde(er_nis_unst).evaluate(x), label='Opposite map - Unstable', linewidth=2)
    plt.plot(x, stats.gaussian_kde(er_nis_stab).evaluate(x), label='Opposite map - Stable', linewidth=2)
    plt.legend()
    plt.xlim([0,1])
    plt.xlabel('position error')
    plt.savefig(path+'_gkde_nIS_%u.pdf' % idx)
    plt.close()

    # histograms
    plt.hist(er_is_unst, label='Internal Map - Unstable', alpha=0.4, weights=np.ones(15*DT)/(15*DT))
    plt.hist(er_is_stab, label='Internal Map - Stable', alpha=0.4, weights=np.ones(15*DT)/(15*DT))
    plt.hist(er_nis_unst, label='Opposite Map - Unstable', alpha=0.4, weights=np.ones(15*DT)/(15*DT))
    plt.hist(er_nis_stab, label='Opposite Map - Stable', alpha=0.4, weights=np.ones(15*DT)/(15*DT))
    plt.legend()
    plt.xlim([0, 1])
    plt.xlabel('position error')
    plt.savefig(path + '_hist_%u.pdf' % idx)
    plt.close()

    np.savetxt(path+'/is_unst.txt', er_is_unst)
    np.savetxt(path + '/is_stab.txt', er_is_stab)
    np.savetxt(path + '/nis_unst.txt', er_nis_unst)
    np.savetxt(path + '/nis_stab.txt', er_nis_stab)

    return er_is_unst, er_is_stab, er_nis_unst, er_nis_stab, er_na_unst, er_na_stab

def position_time_analysis(N, gm, gv, gf, beta, DT=32*20):
    path = '../Plots/Analysis/position_analysis/N%u_M%.2f_V%.2f' % (N, gm, gv)
    mkdir(path)
    er_is_unst = []
    er_is_stab = []
    er_nis_unst = []
    er_nis_stab = []
    er_na_unst = []
    er_na_stab = []
    er_nna_unst = []
    er_nna_stab = []
    T0 = 100

    C = network.CA3(N, 2, 2, alpha_visual=32, alpha_mec=32, alpha_feedback=32, mec_sparsity=0, vis_sparsity=0)
    for j in range(15):
        t = teletimes[j]
        print 'analysis of teleportation %u' % (j)
        C = network.CA3(N, 2, 2, alpha_visual=32, alpha_mec=32, alpha_feedback=32, mec_sparsity=0, vis_sparsity=0)
        if j % 2 == 1:
            C.M = 'B'
            M = 'B'
        elif j % 2 == 0:
            C.M = 'A'
            M = 'A'
        pos = positions[t - T0:t + DT]
        light = light_conditions[t - T0:t + DT]
        C.evolve(light, pos, mec_gamma=gm, vis_gamma=gv, feedback_gamma=gf, beta=beta)
        state = C.population.saved_status

        posA, posB = infer_positions(state, C.population.network.environments[0], C.population.network.environments[1])
        DE = DL(C)

        IS = DE < 0  # 0 for A and 1 for B
        pos_IS = np.where(np.transpose(repmat(IS, 2, 1)), posB, posA)
        pos_nIS = np.where(np.transpose(repmat(IS, 2, 1)), posA, posB)
        if M == 'A':
            pos_NA = np.vstack([posA[:T0, :], posB[T0:, :]])
            pos_nNA = np.vstack([posB[:T0, :], posA[T0:, :]])
        if M == 'B':
            pos_NA = np.vstack([posB[:T0, :], posA[T0:, :]])
            pos_nNA = np.vstack([posA[:T0, :], posB[T0:, :]])

        err_IS = pbc_distance(pos, pos_IS)
        err_nIS = pbc_distance(pos, pos_nIS)
        err_NA = pbc_distance(pos, pos_NA)
        err_nNA = pbc_distance(pos, pos_nNA)

        plot_position(pos, pos_IS, path + '/%u.pdf' % j)
        # visualize.visualize_2D(C, path + '/%u' % j, pos, [T0], plotall=False)

        er_is_unst.append(err_IS[T0:])
        er_is_stab.append(err_IS[:T0])
        er_nis_unst.append(err_nIS[T0:])
        er_nis_stab.append(err_nIS[:T0])
        er_na_unst.append(err_NA[T0:])
        er_nna_unst.append(err_nNA[T0:])
        er_na_stab.append(err_NA[:T0])

    plt.figure()
    plot_shaded_error(60*np.asarray(er_is_unst), label='internal map', color=defC[0], winsize=32)
    plot_shaded_error(60*np.asarray(er_nis_unst), label='opposite map', color=defC[1], winsize=32)
    plt.xlabel('theta cycle after teleportation')
    plt.ylabel('positional error')
    plt.ylim([0,25])
    plt.legend()
    plt.tight_layout()
    plt.savefig('../Plots/Analysis/position-coherence.pdf')
    plt.close()

    plt.figure()
    plot_shaded_error(60 * np.asarray(er_na_unst), label='only visual (new map)', color=defC[2], winsize=32)
    plot_shaded_error(60 * np.asarray(er_nna_unst), label='only PI (old map)', color=defC[3], winsize=32)
    plt.xlabel('theta cycle after teleportation')
    plt.ylabel('positional error')
    #plt.ylim([0, 25])
    plt.legend()
    plt.tight_layout()
    plt.savefig('../Plots/Analysis/position-coherence-2.pdf')
    plt.close()

    return np.asarray(er_nis_unst)

def deltaL_analysis(ntrials, N, gm, gv, beta, DT = 600):
    DE_s = []
    DE_u = []
    for i in range(ntrials):
        C = network.CA3(N, 2, 2, alpha_visual=32, alpha_mec=32, alpha_feedback=32, mec_sparsity=0, vis_sparsity=0)
        for j in range(15):
            C.clear()
            t = teletimes[j]
            print 'analysis of teleportation %u %u' % (i, j)
            if j % 2 == 1:
                C.M = 'B'
            elif j % 2 == 0:
                C.M = 'A'
            pos = positions[t - DT:t + DT]
            light = light_conditions[t - DT:t + DT]
            C.evolve(light, pos, mec_gamma=gm, vis_gamma=gv, feedback_gamma=1000, beta=beta)
            DE = DL(C)
            DE_s.append(np.abs(DE[:DT]))
            DE_u.append(np.abs(DE[DT:]))


    DE_s = np.asarray(DE_s)
    DE_u = np.asarray(DE_u)
    print np.mean(DE_s)
    print np.mean(DE_u)
    np.savetxt('../Data/DL_stable', DE_s.flatten())
    np.savetxt('../Data/DL_unstable', DE_u.flatten())

def remap_times(ntrials, N, gm, gv, gf, beta, DT = 600):
    Rs = []
    for i in range(ntrials):
        C = network.CA3(N, 2, 2, alpha_visual=32, alpha_mec=32, alpha_feedback=32, mec_sparsity=0, vis_sparsity=0)
        for j in range(15):
            C.clear()
            t = teletimes[j]
            print 'analysis of teleportation %u %u' % (i, j)
            if j % 2 == 1:
                C.M = 'B'
            elif j % 2 == 0:
                C.M = 'A'
            pos = positions[t - 100:t + DT]
            light = light_conditions[t - 100:t + DT]
            C.evolve(light, pos, mec_gamma=gm, vis_gamma=gv, feedback_gamma=gf, beta=beta)
            if len(C.remap_time):
                Rs.append(C.remap_time[0]-100)
                print Rs

    np.savetxt('../Data/Rs_gf=%.3f' % gf, Rs)

def sojourn_times(ntrials, N, gm, gv, beta, DT=600):
    souj_coherent = []
    souj_uncoherent = []
    for i in range(ntrials):
        C = network.CA3(N, 2, 2, alpha_visual=32, alpha_mec=32, alpha_feedback=32, mec_sparsity=0, vis_sparsity=0)
        for j in range(15):
            C.clear()
            t = teletimes[j]
            print 'analysis of teleportation %u %u' % (i, j)
            if j % 2 == 1:
                C.M = 'B'
            elif j % 2 == 0:
                C.M = 'A'
            pos = positions[t - 100:t + DT]
            light = light_conditions[t - 100:t + DT]
            C.evolve(light, pos, mec_gamma=gm, vis_gamma=gv, feedback_gamma=1000, beta=beta)
            DE = DL(C)
            DL_plot(DE, 32, '/sojourn_analysis/DL/beta=%.1f_N=%u_gm=%.2f_gv=%.2f_t%u_%u' % (beta, N, gm, gv, j, i), 100)
            [sA, sB] = compute_soujourn_times(DE)
            if j % 2 == 1:
                souj_uncoherent = np.concatenate([souj_uncoherent, sB])
                souj_coherent = np.concatenate([souj_coherent, sA])
            elif j % 2 == 0:
                souj_uncoherent = np.concatenate([souj_uncoherent, sA])
                souj_coherent = np.concatenate([souj_coherent, sB])
    sC = souj_coherent/4.
    sU = souj_uncoherent/4.
    mkdir('../Data/souj')
    np.savetxt('../Data/souj/coherent_N%u_gm%.2f_gv%.2f_beta%.2f' % (N, gm, gv, beta) , sC)
    np.savetxt('../Data/souj/uncoherent_N%u_gm%.2f_gv%.2f_beta%.2f' % (N, gm, gv, beta) , sU)
    return [sC, sU]

def sojourn_times_betas(ntrials, N, gm, gv, betas, DT=600):
    scaleU = []
    scaleC = []
    for b in betas:
        [sC, sU] = sojourn_times(ntrials, N, gm, gv, b, DT)
        [l, scale1] = scipy.stats.expon.fit(sU)
        [l, scale2] = scipy.stats.expon.fit(sC)
        scaleU.append(scale1)
        scaleC.append(scale2)
        print scaleU
        print scaleC
    plt.plot(betas, scaleU, label='coherent')
    plt.plot(betas, scaleC, label='uncoherent')
    plt.xlabel('beta')
    plt.ylabel('soujourn time scale')
    plt.title('N=%u, gv=%.2f, gm=%.2f' % (N, gv, gm))
    plt.show()
    return [scaleC, scaleU]

def flicker_time_correlation(ntrials, N, gm, gv, beta, DT=400):
    f = np.zeros((ntrials*15, DT))
    for i in range(ntrials):
        C = network.CA3(N, 2, 2, alpha_visual=32, alpha_mec=32, alpha_feedback=32, mec_sparsity=0, vis_sparsity=0)
        for j in range(15):
            C.clear()
            t = teletimes[j]
            print 'analysis of teleportation %u %u' % (i, j)
            if j % 2 == 1:
                C.M = 'B'
            elif j % 2 == 0:
                C.M = 'A'
            pos = positions[t - 100:t + DT]
            light = light_conditions[t - 100:t + DT]
            C.evolve(light, pos, mec_gamma=gm, vis_gamma=gv, feedback_gamma=1000, beta=beta)
            DE = DL(C)
            DL_plot(DE, 4, 'Diagram/deltaL/%.3f_%.3f_%u' % (gv, beta, j), 100, remap=[])
            if j % 2 == 1:
                f[i*15 + j] = DE[100:] < 0
            elif j % 2 == 0:
                f[i * 15 + j] = DE[100:] > 0
    # c = array_time_correlation(f, 40)
    c = time_correlation(f.flatten(), 40)
    return c


