import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch
import codecs, json
import random

from sketches import hadamard
from generate_dataset import load_data
from solvers_lr import LogisticRegression
from solvers_ls import RidgeRegression

def plot_loss_vs_time(losses, times, sketches, schemes, PARAMS):
    MIN_LOSS = PARAMS['MIN_LOSS']
    MAX_TIME = PARAMS['CROP_TIME']
    OUTPUT_DIR = PARAMS['OUTPUT_DIR']
    MARKERS = PARAMS['MARKERS']
    counter = 0
    for (k,v) in losses.items():
        plt.plot(times[k], v, MARKERS[counter], label=k, linewidth=2)
        counter += 1
    plt.legend()
    plt.title(PARAMS['dataset'] + ' ' + PARAMS['dimensions'])
    plt.xlabel('Wall clock time')
    plt.ylabel('Error')
    plt.yscale('log')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(OUTPUT_DIR + '/loss_vs_time.png')
    plt.ylim([MIN_LOSS, 9])
    plt.xlim([0, MAX_TIME])
    plt.savefig(OUTPUT_DIR + '/loss_vs_time_cropped.png')
    plt.cla()
    

def plot_loss_vs_iters(losses, times, sketches, schemes, PARAMS):
    MIN_LOSS = PARAMS['MIN_LOSS']
    CROP_ITER = PARAMS['CROP_ITER']
    OUTPUT_DIR = PARAMS['OUTPUT_DIR']
    MARKERS = PARAMS['MARKERS']
    counter = 0
    for (k,v) in losses.items():
        plt.plot(np.arange(times[k].shape[0]), v, MARKERS[counter], label=k, linewidth=2)
        counter += 1

    plt.legend()
    plt.title(PARAMS['dataset'] + ' ' + PARAMS['dimensions'])
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.yscale('log')
    # plt.xticks(np.arange(times[k].shape[0]),np.arange(times[k].shape[0]))
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(OUTPUT_DIR + '/loss_vs_iters.png')
    plt.ylim([MIN_LOSS, 9])
    plt.xlim([0, CROP_ITER])
    plt.savefig(OUTPUT_DIR + '/loss_vs_iters_cropped.png')
    
def plot_loss_vs_time_for(losses, times, sketches, schemes, PARAMS):
    MIN_LOSS = PARAMS['MIN_LOSS']
    CROP_ITER = PARAMS['CROP_ITER']
    OUTPUT_DIR = PARAMS['OUTPUT_DIR']
    MARKERS = PARAMS['MARKERS']
    counter = 0
    for (k,v) in losses.items():
        plt.plot(np.arange(times[k].shape[0]), v, MARKERS[counter], label=k, linewidth=2)
        counter += 1

    plt.legend()
    plt.title(PARAMS['dataset'] + ' ' + PARAMS['dimensions'])
    plt.xlabel('Wall clock time')
    plt.ylabel('Error')
    plt.yscale('log')
    # plt.xticks(np.arange(times[k].shape[0]),np.arange(times[k].shape[0]))
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(OUTPUT_DIR + '/loss_vs_iters.png')
    plt.ylim([MIN_LOSS, 9])
    plt.xlim([0, CROP_ITER])
    plt.savefig(OUTPUT_DIR + '/loss_vs_iters_cropped.png')


def save_results(losses, times, PARAMS):
    OUTPUT_DIR = PARAMS['OUTPUT_DIR']
    losses = {k:v.tolist() for (k,v) in losses.items()}
    times = {k:v.tolist() for (k,v) in times.items()}
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    json.dump(losses,codecs.open(OUTPUT_DIR + '/losses.json', 'w', encoding='utf-8'))
    json.dump(times,codecs.open(OUTPUT_DIR + '/times.json', 'w', encoding='utf-8'))

def load_results(PARAMS):
    OUTPUT_DIR = PARAMS['OUTPUT_DIR']
    losses = {}
    times = {}
    if os.path.exists(OUTPUT_DIR):
        losses_text = codecs.open(OUTPUT_DIR + '/losses.json', 'r', encoding='utf-8').read()
        losses = json.loads(losses_text)
        times_text = codecs.open(OUTPUT_DIR + '/times.json', 'r', encoding='utf-8').read()
        times = json.loads(times_text)
    losses = {k:np.array(v) for (k,v) in losses.items()}
    times = {k:np.array(v) for (k,v) in times.items()}
    return losses, times
        
    
def main():
    if len(sys.argv) < 8:
        print("usage: main.py <dataset> <experiment> <num datapoints> <num random features> <sketch size> <num trials> <num machines>")
        exit(1)

    # declaring variables
    dataset = sys.argv[1]
    experiment = sys.argv[2]
    n = int(sys.argv[3]) # num datapoints
    d = int(sys.argv[4]) # num features
    m = int(sys.argv[5]) # sketch size
    n_trials = int(sys.argv[6])
    k = int(sys.argv[7]) # num machine
    if d > n:
        print("num features should be less than num datapoints. " + \
        "Will continue running but this setup doesn't make a lot of sense.")
    if m >= n:
        print("sketch size is not smaller than original matrix. " + \
        "Will continue running but this setup doesn't make a lot of sense.")
    
    sketches = ['rrs']
    schemes = ['unweighted']
    lambd = 1e-8 # 1e-8
    MAX_TIME = 400
    MIN_LOSS = 1e-15
    PARAMS = {"dataset" : dataset, "MAX_TIME" : MAX_TIME, "CROP_TIME" : 30, "CROP_ITER" : 23, "MIN_LOSS" : 2e-8, "n_trials" : n_trials, "dimensions" : '( d=' + str(d) + ' )'}
    if dataset.startswith("Synthetic") or dataset.startswith("High Coherence"):
        PARAMS['dimensions'] = '( n=' + str(int(n/1000)) + 'k, d=' + str(d) + ' )'
    PARAMS["OUTPUT_DIR"] = "output" + '_' + experiment + '/' + PARAMS["dataset"] + '_n='+str(n)+'_d='+str(d)+'_m='+str(m)+'_m='+str(k)
    PARAMS["MARKERS"] = ['b-','y:','r--','g-.','o-','^-','>-','<-']
    if experiment == "least_squares_averaging":
        PARAMS["MARKERS"] = ['r-','g-','r--','g--','r-.','g-.','r:','g:']
    elif experiment == "svrn":
        PARAMS["MARKERS"] = ['b-','y:','r--','g-.','y:','o-','^-','>-','<-']
    elif experiment == "dsvrn":
        PARAMS["MARKERS"] = ['b-','y:','r--','g-.','y:','o-','^-','>-','<-']
    elif experiment == "least_squares_coherence":
        PARAMS["MARKERS"] = ['r-','g-','r--','g--','^-','>-','<-']
    elif experiment == "least_squares_sampling":
        PARAMS["MARKERS"] = ['r--','y:','g-.','b-']
    elif experiment =="least_squares_vr":
        PARAMS["MARKERS"] = ['r--','y:','g-.','b-']
        
    kappa = 10



    if n_trials > 0:
        A, b = load_data(dataset=dataset,n=n, d=d, kappa=kappa,high_coherence=False)
        ########################################
        # number of machine
        # round the number of data points
        true_dim = A.shape[0] - A.shape[0] % k
        # true_dim1 = A.shape[0] - A.shape[0] % 2
        A = A[:true_dim,:]
        # A1 = A[:true_dim1,:]
        # random shuffle the data once here
        randidx = torch.randperm(A.shape[0])
        A = A[randidx]
        # A1 = A1[randidx]
        b = b[randidx]
        
        ########################################
        n, d = A.shape
        # n1, d1 = A1.shape
        nnz = d/n
        # nnz1 = d1/n1
    
        n_iter_newton = 20
        n_iter_svrg = 25
        n_iter_svrn = 20
        n_iter_dsvrn = 20
        n_iter_ihs = 25
    
        losses = {}
        times = {}
        times_for = {}

        # preprocessing step
        # if experiment.startswith("least_squares"):
        lreg = RidgeRegression(A, b, lambd, MAX_TIME, MIN_LOSS, k)
        _ = lreg.solve_exactly()
        # else:
        # lreg = LogisticRegression(A, b, lambd, MAX_TIME, MIN_LOSS, k)
        # _, _ = lreg.solve_exactly(n_iter=20, eps=1e-16)
        # lreg1 = LogisticRegression(A, b, lambd, MAX_TIME, MIN_LOSS, 2)
        # _, _ = lreg1.solve_exactly(n_iter=20, eps=1e-16)
        # lreg2 = LogisticRegression(A, b, lambd, MAX_TIME, MIN_LOSS, 3)
        # _, _ = lreg2.solve_exactly(n_iter=20, eps=1e-16)

        if experiment == "svrn":
            sketch = sketches[0]
            scheme = schemes[0]
            NEWTON = "Newton"
            SN = "SN-HA"
            SVRN = "SVRN-HA"
            DSVRN = "DSVRN-HA"
            losses[NEWTON] = np.zeros(n_iter_newton+1)
            times[NEWTON] = np.zeros(n_iter_newton+1)
            losses[SN] = np.zeros(n_iter_ihs+1)
            times[SN] = np.zeros(n_iter_ihs+1)
            losses[SVRN] = np.zeros(n_iter_svrn+1)
            times[SVRN] = np.zeros(n_iter_svrn+1)
            losses[DSVRN] = np.zeros(n_iter_dsvrn+1)
            times[DSVRN] = np.zeros(n_iter_dsvrn+1)
            

            for _ in range(n_trials):
                print("trial number " + str(_) + '\n')
                # _, losses_newton, times_newton = lreg.newton(n_iter_newton)
                # losses[NEWTON] += losses_newton / n_trials
                # times[NEWTON] += times_newton / n_trials
                # _, losses_, times_ = lreg.ihs(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_ihs, scheme=scheme)
                # losses[SN] += losses_ / n_trials
                # times[SN] += times_ / n_trials
                # _, losses_svrn, times_svrn = lreg.ihs_svrn(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_svrn,scheme=scheme)
                # losses[SVRN] += losses_svrn / n_trials
                # times[SVRN] += times_svrn / n_trials


                _, losses_dsvrn, times_dsvrn = lreg.ihs_dsvrn(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_dsvrn,scheme=scheme)
                losses[DSVRN] += losses_dsvrn / n_trials
                times[DSVRN] += times_dsvrn / n_trials
        
            
            print(losses[SVRN])
            print(losses[DSVRN])

        # Calling DSVRN
        elif experiment == "dsvrn":
            sketch = sketches[0]
            scheme = schemes[0]
            # NEWTON = "Newton"
            # SN = "SN-HA"
            SVRN = "SVRN-HA"
            DSVRN = "DSVRN-HA"
            # DSVRN1 = "DSVRN-HA with m=2"
            # DSVRN2 = "DSVRN-HA with m=3"
            # losses[NEWTON] = np.zeros(n_iter_newton+1)
            # times[NEWTON] = np.zeros(n_iter_newton+1)
            # losses[SN] = np.zeros(n_iter_ihs+1)
            # times[SN] = np.zeros(n_iter_ihs+1)
            losses[SVRN] = np.zeros(n_iter_svrn+1)
            times[SVRN] = np.zeros(n_iter_svrn+1)
            losses[DSVRN] = np.zeros(n_iter_dsvrn+1)
            times[DSVRN] = np.zeros(n_iter_dsvrn+1)
            # losses[DSVRN1] = np.zeros(n_iter_dsvrn+1)
            # times[DSVRN1] = np.zeros(n_iter_dsvrn+1)
            # losses[DSVRN2] = np.zeros(n_iter_dsvrn+1)
            # times[DSVRN2] = np.zeros(n_iter_dsvrn+1)

            for _ in range(n_trials):
                print("trial number " + str(_) + '\n')
                # _, losses_newton, times_newton = lreg.newton(n_iter_newton)
                # losses[NEWTON] += losses_newton / n_trials
                # times[NEWTON] += times_newton / n_trials
                # _, losses_, times_ = lreg.ihs(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_ihs, scheme=scheme)
                # losses[SN] += losses_ / n_trials
                # times[SN] += times_ / n_trials
                
                lreg = RidgeRegression(A, b, lambd, MAX_TIME, MIN_LOSS, k)
                _ = lreg.solve_exactly()
                _, losses_svrn, times_svrn = lreg.ihs_svrn(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_svrn,scheme=scheme)
                losses[SVRN] += losses_svrn / n_trials
                times[SVRN] += times_svrn / n_trials
                
                lreg = RidgeRegression(A, b, lambd, MAX_TIME, MIN_LOSS, k)
                _ = lreg.solve_exactly()
                _, losses_dsvrn, times_dsvrn, elapsed_time = lreg.ihs_dsvrn(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_svrn,scheme=scheme)
                losses[DSVRN] += losses_dsvrn / n_trials
                times[DSVRN] += (times_dsvrn - elapsed_time) / n_trials
                
                # _, losses_dsvrn, times_dsvrn, _ = lreg1.ihs_dsvrn1(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_dsvrn,scheme=scheme)
                # losses[DSVRN1] += losses_dsvrn / n_trials
                # times[DSVRN1] += (times_dsvrn) / n_trials
                
                # _, losses_dsvrn, times_dsvrn, _ = lreg2.ihs_dsvrn(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_dsvrn,scheme=scheme)
                # losses[DSVRN1] += losses_dsvrn / n_trials
                # times[DSVRN1] += (times_dsvrn) / n_trials
                
                
                # _, losses_dsvrn, times_dsvrn, for_loop_times_dsvrn  = lreg.ihs_dsvrn(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_svrn,scheme=scheme)
                # losses[DSVRN1] += losses_dsvrn / n_trials
                # times[DSVRN1] += (times_dsvrn - for_loop_times_dsvrn) / n_trials
                
                
            
                
        elif experiment == "svrg":
            steps = list(map(float,sys.argv[7].split())) if len(sys.argv) >= 8 else [0.03, 0.1, 0.3, 1.0]
            sizes = [m,2*m]
            print(steps)
            for _ in range(n_trials):
                print("trial number " + str(_) + '\n')
                for step_ in steps:
                    for size_ in sizes:
                        name = "svrg (step=" + str(step_)+", m="+str(size_)+")"
                        if name not in losses:
                            losses[name] = np.zeros(n_iter_svrg+1)
                            times[name] = np.zeros(n_iter_svrg+1)
                        _, losses_svrg, times_svrg = lreg.svrg(size_, n_iter=n_iter_svrg,s=step_,batch_size=10)
                        losses[name] += losses_svrg / n_trials
                        times[name] += times_svrg / n_trials
        elif experiment == "least_squares_sampling":
            sketch = sketches[0]
            scheme = schemes[0]
            SN = sketch+' '+scheme
            samplings = ['once', 'per stage', 'per step']
            losses[SN] = np.zeros(n_iter_ihs+1)
            times[SN] = np.zeros(n_iter_ihs+1)
            for sampling in samplings:
                SVRN = "svrn (sampling "+sampling+')'
                losses[SVRN] = np.zeros(n_iter_svrn+1)
                times[SVRN] = np.zeros(n_iter_svrn+1)
            for _ in range(n_trials):
                print("trial number " + str(_) + '\n')
                _, losses_, times_ = lreg.ihs(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_ihs, scheme=scheme)
                losses[SN] += losses_ / n_trials
                times[SN] += times_ / n_trials
                for sampling in samplings:
                    SVRN = "svrn (sampling "+sampling+')'
                    _, losses_svrn, times_svrn = lreg.ihs_svrn(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_svrn,scheme=scheme,sampling=sampling)
                    losses[SVRN] += losses_svrn / n_trials
                    times[SVRN] += times_svrn / n_trials
        elif experiment == "least_squares_vr":
            sketch = sketches[0]
            scheme = schemes[0]
            SN = 'SN-HA'
            SNGS = 'SNGS-HA'
            SVRN = 'SVRN-HA'
            losses[SN] = np.zeros(n_iter_ihs+1)
            times[SN] = np.zeros(n_iter_ihs+1)
            losses[SVRN] = np.zeros(n_iter_svrn+1)
            times[SVRN] = np.zeros(n_iter_svrn+1)
            losses[SNGS] = np.zeros(n_iter_svrn+1)
            times[SNGS] = np.zeros(n_iter_svrn+1)
            for _ in range(n_trials):
                print("trial number " + str(_) + '\n')
                _, losses_, times_ = lreg.ihs(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_ihs, scheme=scheme)
                losses[SN] += losses_ / n_trials
                times[SN] += times_ / n_trials
                _, losses_svrn, times_svrn = lreg.ihs_svrn(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_svrn,scheme=scheme,sampling='per step',with_vr=False)
                losses[SNGS] += losses_svrn / n_trials
                times[SNGS] += times_svrn / n_trials
                _, losses_svrn, times_svrn = lreg.ihs_svrn(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_svrn,scheme=scheme,sampling='per step',with_vr=True)
                losses[SVRN] += losses_svrn / n_trials
                times[SVRN] += times_svrn / n_trials
        elif experiment == "least_squares_noaveraging":
            sketch = 'rrs'
            scheme = 'unweighted'
            sizes = [m,2*m,3*m]
            for size in sizes:
                SVRN = "SVRN (h=" + str(size) + ")"
                SN = "SN (h=" + str(size) + ")"
                losses[SN] = np.zeros(n_iter_ihs+1)
                times[SN] = np.zeros(n_iter_ihs+1)
                losses[SVRN] = np.zeros(n_iter_svrn+1)
                times[SVRN] = np.zeros(n_iter_svrn+1)
            for _ in range(n_trials):
                print("trial number " + str(_) + '\n')
                for size in sizes:
                    SVRN = "SVRN (h=" + str(size) + ")"
                    SN = "SN (h=" + str(size) + ")"
                    _, losses_, times_ = lreg.ihs(sketch_size=size,sketch=sketch, nnz=nnz, n_iter=n_iter_ihs, scheme=scheme,stop_averaging=1)
                    losses[SN] += losses_ / n_trials
                    times[SN] += times_ / n_trials
                    _, losses_svrn, times_svrn = lreg.ihs_svrn(sketch_size=size,sketch=sketch, nnz=nnz, n_iter=n_iter_svrn,scheme=scheme,sampling='per step',stop_averaging=1)
                    losses[SVRN] += losses_svrn / n_trials
                    times[SVRN] += times_svrn / n_trials
        elif experiment == "least_squares_coherence":
            sketch = 'rrs'
            scheme = 'unweighted'
            SN = 'SN-HA'
            SNlev = 'SN-HA-RHT'
            SVRN = 'SVRN-HA'
            SVRNlev = 'SVRN-HA-RHT'
            losses[SN] = np.zeros(n_iter_ihs+1)
            times[SN] = np.zeros(n_iter_ihs+1)
            losses[SVRN] = np.zeros(n_iter_svrn+1)
            times[SVRN] = np.zeros(n_iter_svrn+1)
            losses[SNlev] = np.zeros(n_iter_ihs+1)
            times[SNlev] = np.zeros(n_iter_ihs+1)
            losses[SVRNlev] = np.zeros(n_iter_svrn+1)
            times[SVRNlev] = np.zeros(n_iter_svrn+1)
            for _ in range(n_trials):
                print("trial number " + str(_) + '\n')
                _, losses_, times_ = lreg.ihs(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_ihs, scheme=scheme)
                losses[SN] += losses_ / n_trials
                times[SN] += times_ / n_trials
                _, losses_svrn, times_svrn = lreg.ihs_svrn(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_svrn,scheme=scheme,sampling='per step')
                losses[SVRN] += losses_svrn / n_trials
                times[SVRN] += times_svrn / n_trials

            lreg.A, v = hadamard(lreg.A, return_diag=True)
            lreg.n, lreg.d = lreg.A.shape
            lreg.b = hadamard(lreg.b, diag = v)
    
            for _ in range(n_trials):
                print("trial number " + str(_) + '\n')
                _, losses_, times_ = lreg.ihs(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_ihs, scheme=scheme)
                losses[SNlev] += losses_ / n_trials
                times[SNlev] += times_ / n_trials
                _, losses_svrn, times_svrn = lreg.ihs_svrn(sketch_size=m,sketch=sketch, nnz=nnz, n_iter=n_iter_svrn,scheme=scheme,sampling='per step',permanent_switch=True)
                losses[SVRNlev] += losses_svrn / n_trials
                times[SVRNlev] += times_svrn / n_trials
            

            

    else:
        losses, times = load_results(PARAMS)


    times = {k:v[0:np.argmax(v)+1] for (k,v) in times.items()}
    losses = {k:v[0:len(times[k])] for (k,v) in losses.items()}
    times_for = {k:v[0:np.argmax(v)+1] for (k,v) in times_for.items()}
        
    # plotting
    plot_loss_vs_time(losses, times, sketches, schemes, PARAMS)
    plot_loss_vs_iters(losses, times, sketches, schemes, PARAMS)
    # plot_loss_vs_time_for(losses, times_for, sketches, schemes, PARAMS)

    if n_trials > 0:
        save_results(losses, times, PARAMS)




        
if __name__ == '__main__':
    main()