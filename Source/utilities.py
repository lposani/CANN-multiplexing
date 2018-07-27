from imports import *

def zero_to_nan(values):
    return [[float('nan') if x==0 else x for x in d] for d in values]

def mkdir(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)

def smooth(x, window_len=15, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

def dist(x,y):
    return np.sqrt(np.sum((x-y) * (x-y),1))

def pbc_distance(x, y):
    d = dist(x,y)
    for xpbc in [-1, 0, 1]:
        for ypbc in [-1, 0, 1]:
            dp = dist(x, y + [xpbc, ypbc])
            d = np.minimum(d, dp)
    return d

def sign_changes(DE, k, mode):
    smoothDE = smooth(DE, k)
    smoothDE = smoothDE[k-1:-k+1]
    if mode=='A':
        sDE = (smoothDE[1:] * smoothDE[:-1] < 0) * (smoothDE[:-1] < 0)
    if mode == 'B':
        sDE = (smoothDE[1:] * smoothDE[:-1] < 0) * (smoothDE[:-1] > 0)
    s = np.sum(sDE)
    print s
    return s

def flickering_rate_time(DE, remaps, M, dt, perc=0):
    dt = int(dt)
    if M=='B':
        DE*=-1
        perc*= -1
    T = len(DE)
    rates = []
    alive = []
    times = np.linspace(dt,T,T/dt)
    for t in times:
        t = int(t)
        #print np.sum(DE[t-dt:t] < 0)/float(dt)
        rates.append(np.sum(DE[t-dt:t] < perc)/float(dt))
        if len(remaps)>0 and t>remaps[0]:
            alive.append(np.sum((np.linspace(t-dt, t, dt)<remaps[0])/float(dt)))
        else:
            alive.append(1)
    return np.asarray(rates), np.asarray(alive)

def permanence_times(DE):
    A = float(np.sum(DE>0))/len(DE)
    B = float(np.sum(DE < 0))/len(DE)
    return (A,B)

def DL(C):
    state = C.population.saved_status
    T = len(state)
    EA = np.zeros(T)
    EB = np.zeros(T)
    for k in range(T):
        EA[k] = -0.5 * np.dot(state[k], np.dot(C.population.network.Js[0], state[k]))
        EB[k] = -0.5 * np.dot(state[k], np.dot(C.population.network.Js[1], state[k]))
    DE = EB - EA
    return DE

def DL_plot(DE, dt, title, telep, remap=[]):
    dt = float(dt)
    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(len(DE))/dt, DE, 1/dt)
    plt.xlabel('time')
    plt.ylabel('$\Delta$L')
    y = plt.ylim()
    plt.plot([telep/dt, telep/dt], y, 'r')
    if len(remap)>0:
        for r in remap:
            plt.plot([r/dt,r/dt], y, 'g')
    plt.ylim(y)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('../Plots/'+title+'.pdf')
    plt.close()

def infer_positions(state, env_A, env_B):
    posA = []
    posB = []
    for s in state:
        x = env_A[s > 0, 0]
        y = env_A[s > 0, 1]
        xA = np.arctan2(-(np.mean(np.sin(2 * np.pi * x))) , -(np.mean(np.cos(2 * np.pi * x))))/(2*np.pi) + 0.5
        yA = np.arctan2(-(np.mean(np.sin(2 * np.pi * y))) , -(np.mean(np.cos(2 * np.pi * y))))/(2*np.pi) + 0.5

        x = env_B[s > 0, 0]
        y = env_B[s > 0, 1]
        xB = np.arctan2(-(np.mean(np.sin(2 * np.pi * x))) , -(np.mean(np.cos(2 * np.pi * x))))/(2*np.pi) + 0.5
        yB = np.arctan2(-(np.mean(np.sin(2 * np.pi * y))) , -(np.mean(np.cos(2 * np.pi * y))))/(2*np.pi) + 0.5
        posA.append([xA, yA])
        posB.append([xB, yB])
    return np.asarray(posA), np.asarray(posB)

def plot_position(pos, pos_IS, savepath):
    err_IS = pbc_distance(pos, pos_IS)
    plt.figure(figsize=(12,6))
    t = np.linspace(0,len(pos), len(pos))/32.
    plt.subplot(3,1,1)
    plt.plot(t, pos_IS[:,0],label='inferred position')
    plt.plot(t, pos[:,0], '--k', label='true position')
    plt.legend()
    plt.ylabel('x coordinate')
    plt.ylim([0,1])
    plt.xlim([0,len(pos)/32.])
    plt.subplot(3, 1, 2)
    plt.plot(t, pos_IS[:, 1], label='inferred position')
    plt.plot(t, pos[:, 1], '--k', label='true position')
    plt.ylabel('y coordinate')
    plt.ylim([0, 1])
    plt.xlim([0, len(pos)/32.])
    plt.subplot(3,1,3)
    plt.bar(t, err_IS)
    plt.ylim([0, 0.5*np.sqrt(2)])
    plt.xlim([0, len(pos)/32.])
    plt.ylabel('positional error')
    plt.savefig(savepath)
    plt.close()

def plot_shaded_error(ys, ax=0, label=None, color=None, winsize=1):
    d = np.shape(ys)
    n = d[ax]
    t = d[1-ax]
    print n, t
    ys = np.reshape(np.transpose(ys), (t/winsize, n*winsize))
    print ys.shape
    y =  np.mean(ys, ax-1)
    err = np.std(ys, ax-1)

    #y = smooth(y, 80)
    #err = smooth(err, 80)

    down = y-err/np.sqrt(n*winsize)
    x = np.linspace(0,t/4., len(y)+1)
    x = x[1:]
    up = y+err/np.sqrt(n*winsize)

    if color:
        plt.plot(x, y, color=color)
        plt.fill_between(x, down, up, label=label, color=color, alpha=0.3)
    else:
        plt.plot(x,y, 'k')
        plt.fill_between(x, down, up, label=label)

def font_size(s):
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(s)

def compute_soujourn_times(DE):
    smoothDE = DE
    sA = []
    sB = []
    l = 0
    for t in range(1, len(smoothDE)):
        l+=1
        if smoothDE[t] * smoothDE[t-1] < 0:
            if smoothDE[t]<0:
                sA.append(l)
            elif smoothDE[t]>0:
                sB.append(l)
            l=0
    return [sA, sB]

def temporal_correlation(F, t0, tmax):
    tmax = tmax+1
    mu1 = np.mean(F[:-tmax])
    mu2 = np.mean(F[t0:-tmax+t0])
    C = np.mean(F[:-tmax] * F[t0:-tmax+t0])
    return C - mu1*mu2

def time_correlation(F, tmax):
    C=np.zeros(tmax)
    for t in range(1,tmax+1):
        C[t-1] = temporal_correlation(F, t, tmax)
    return C

def array_time_correlation(F, tmax):
    n = np.shape(F)[0]
    C = np.zeros(tmax)
    for f in F:
        C += time_correlation(f, tmax)/n
    return C

def load_positions(steps_per_theta):
    x = np.loadtxt('../Data/positions_x_%u.txt' % steps_per_theta)
    y = np.loadtxt('../Data/positions_y_%u.txt' % steps_per_theta)
    positions = np.transpose(np.asarray([x, y]))
    return positions

def load_light_conditions(steps_per_theta):
    teletimes = steps_per_theta * np.loadtxt('../Data/tele_times_theta.txt', dtype='int')
    x = np.loadtxt('../Data/positions_x_%u.txt' % steps_per_theta)
    y = np.loadtxt('../Data/positions_y_%u.txt' % steps_per_theta)
    positions = np.transpose(np.asarray([x, y]))
    T = len(positions)
    light_conditions = np.zeros(T)
    for i in range(7):
        light_conditions[teletimes[2 * i]:teletimes[2 * i + 1]] = 1
    light_conditions = np.asarray(light_conditions, dtype='int')
    return light_conditions

def load_teletimes(steps_per_theta):
    teletimes = steps_per_theta * np.loadtxt('../Data/tele_times_theta.txt', dtype='int')
    return teletimes