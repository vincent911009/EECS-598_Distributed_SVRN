from re import X
import numpy as np
import torch
from time import time
import random
from sketches import gaussian, less, sparse_rademacher, srht, rrs, rrs_lev_scores, lev_approx
from joblib import Parallel, delayed
from joblib import parallel_backend
# g0_new = None
# L_new = None

SKETCH_FN = {'gaussian': gaussian, 'less': less, 'less_sparse': sparse_rademacher,
             'srht': srht, 'rrs': rrs, 'rrs_lev_scores': rrs_lev_scores}


torch.set_default_dtype(torch.float64)

class RidgeRegression:
    
    def __init__(self, A, b, lambd, MAX_TIME, MIN_LOSS, machine):
        self.A = A
        self.b = b
        if self.b.ndim == 1:
            self.b = self.b.reshape((-1,1))
        self.n, self.d = A.shape
        self.c = self.b.shape[1]
        self.lambd = lambd
        self.device = A.device
        self.MIN_LOSS = MIN_LOSS
        self.MAX_TIME = MAX_TIME
        self.stop_averaging = 0
        ########################################
        self.m = machine
        self.local_A = self.A.view(self.m, int(self.n / self.m), -1)
        self.local_b = self.b.view(self.m, int(self.n / self.m), -1)
        self.local_data_size = int(self.n / self.m)
        ########################################


        
    def loss(self, x):
        return 1./2 * ((self.H_opt @ (x - self.x_opt))**2).sum()
    
    def square_loss(self, x):
        return ((self.A @ x - self.b)**2).mean() + self.lambd/2 * (x**2).sum()

    def grad(self, x, indices=[]):
        batch_size = len(indices)
        if batch_size>1:
            return 1./batch_size*self.A[indices,::].T @ (self.A[indices,::] @ x - self.b[indices])+ self.lambd * x
        elif batch_size==0:
            return 1./self.n*self.A.T @ (self.A @ x - self.b)+ self.lambd * x
        else:
            index = indices[0]
            # index = np.random.choice(self.n)
            a_i = self.A[index,::].reshape((-1,1))
            b_i = self.b[index].squeeze()
            return a_i.reshape((-1,1)) * ((a_i*x).sum() - b_i)+ self.lambd * x
    
    def dgrad(self, x, m_num, indices=[]):
        batch_size = len(indices)
        local_data_size = self.local_data_size
        if self.local_A[m_num].shape[0] != local_data_size or self.local_b[m_num].shape[0] != local_data_size:
          print("wrong local data size!!")
        if batch_size>1:
            return 1./batch_size * self.local_A[m_num][indices,::].T \
             @ (self.local_A[m_num][indices,::] @ x - self.local_b[m_num][indices]) + self.lambd * x
        elif batch_size==0:
            # print('x:',x.shape)
            # print('A:',self.local_A[m_num].shape)
            # print('b:',self.local_b[m_num].shape)
            # full batch gradient computation
            return 1./local_data_size * self.local_A[m_num].T @ (self.local_A[m_num] @ x - self.local_b[m_num]) + self.lambd * x
        else:
            index = indices[0]
            # index = np.random.choice(self.n)
            a_i = self.local_A[m_num][index,::].reshape((-1,1))
            b_i = self.local_b[m_num][index].squeeze()
            return a_i.reshape((-1,1)) * ((a_i*x).sum() - b_i)+ self.lambd * x
    
    def hessian(self, x):
        return 1./self.n * self.A.T @ self.A + self.lambd * torch.eye(self.d).to(self.device)

    def sqrt_hess(self, x):
        return 1./np.sqrt(self.n)

    def line_search(self, x, v, g, alpha=0.3, beta=0.8):
        delta = (v*g).sum()
        loss_x = self.square_loss(x)
        s = 1
        xs = x + s*v
        ls_passes = 1
        while self.square_loss(xs) > loss_x + alpha*s*delta and ls_passes<50:
            s = beta*s
            xs = x + s*v
            ls_passes += 1
        print("line search: steps = " + str(ls_passes) + ", step size = " + str(s))
        return s


    def uniform_weight(self, iter_num):
        return iter_num
    
    def poly_weight(self, iter_num, p=2):
        return (iter_num) ** p

    def log_weight(self, iter_num):
        return (iter_num) ** np.log(iter_num+2)
    
    
    def solve_exactly(self):
        x = torch.zeros(self.d,self.c).to(self.device)
        
        g = self.grad(x)
        H = self.hessian(x)
        u = torch.linalg.cholesky(H)
        v = -torch.cholesky_solve(g, u)
        x = x + v
        
        self.x_opt = x
        self.H_opt = H
        
        _, sigma, _ = torch.svd(self.H_opt)
        
        de_ = torch.trace((H - self.lambd * torch.eye(self.d).to(self.device)) @ torch.pinverse(H))
        self.de = de_.cpu().numpy().item()
        
        return x
    
    def ihs_no_averaging(self, x, sketch_size, sketch, nnz):
        
        start = time()
        
        hsqrt = self.sqrt_hess(x)
        sa = SKETCH_FN[sketch](hsqrt * self.A, sketch_size, nnz=nnz)
        g = self.grad(x)
        
        if sketch_size >= self.d:
            hs = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
            u = torch.linalg.cholesky(hs)
            v = -torch.cholesky_solve(g, u)
        elif sketch_size < self.d:
            ws = sa @ sa.T + self.lambd * torch.eye(sketch_size).to(self.device)
            u = torch.linalg.cholesky(ws)
            sol_ = torch.cholesky_solve(sa @ g, u)
            v = -1./self.lambd * (g - sa.T @ sol_)
            
        s = self.line_search(x, v, g)
        x = x + s*v
        
        return x, time()-start


    def ihs_unweighted_(self, x, sketch_size, sketch, nnz, hs_old, iter):
        start = time()
        
        g = self.grad(x)

        if self.stop_averaging == 0 or iter < self.stop_averaging:
            hsqrt = self.sqrt_hess(x, 1)
            sa = SKETCH_FN[sketch](hsqrt * self.A, sketch_size, nnz=nnz)

            w_old = self.uniform_weight(iter)
            w = self.uniform_weight(iter + 1)
            hs_ = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
            hs = (w_old / w) * hs_old + (1 - w_old / w) * hs_
        else:
            hs = hs_old
            
        u = torch.linalg.cholesky(hs)
        v = -torch.cholesky_solve(g, u)
        # v = -torch.pinverse(hs) @ g
        
        s = self.line_search(x, v, g)
        x = x + s*v
                
        return x, hs, time()-start

    
    def ihs(self, sketch_size, sketch='gaussian', nnz=1., n_iter=10, scheme='unweighted',stop_averaging=0):
        self.stop_averaging = stop_averaging
        
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)
        hs = torch.zeros(self.d,self.d).to(self.device)

        print("IHS "+scheme + "\n")
        losses = [self.loss(x).cpu().numpy().item()]
        times = [0.]
        for i in range(n_iter):
            print("Pass "+str(i))
            if scheme == 'unweighted':
                x, hs, time_ = self.ihs_unweighted_(x, sketch_size, sketch, nnz, hs, i)
            elif scheme == 'poly':
                x, hs, time_ = self.ihs_poly_(x, sketch_size, sketch, nnz, hs, i)
            elif scheme == 'log':
                x, hs, time_ = self.ihs_log_(x, sketch_size, sketch, nnz, hs, i)
            else:
                x, time_ = self.ihs_no_averaging(x, sketch_size, sketch, nnz)
            losses.append(self.loss(x).cpu().numpy().item())
            times.append(time_)
            if np.sum(times) > self.MAX_TIME or losses[-1] < self.MIN_LOSS:
                losses = np.append(losses, np.zeros(n_iter - (i + 1)) + losses[-1])
                times = np.append(times, np.zeros(n_iter - (i + 1)))
                break
        
        losses = np.array(losses)
        times = np.array(times)
        losses /= losses[0]
        
        return x, losses, np.cumsum(times)


    def ihs_svrn(self, sketch_size, sketch='rrs', nnz=.1, n_local = 0, n_iter=10, scheme='unweighted', sampling='per stage',with_vr=True,s=1,stop_averaging=0,permanent_switch=False):
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)
        hs = torch.zeros(self.d,self.d).to(self.device)

        A_full = self.A
        b_full = self.b
        n_full = self.n


        
        losses = [self.loss(x).cpu().numpy().item()]
        times = [0.]
        if n_local == 0:
            n_local = max(int(np.log(self.n/self.d)/np.log(2)),2)

        batch_size = int(self.n / n_local)
        print("SVRN with batch_size=" + str(batch_size) + " and n_local=" + str(n_local)+"\n")

        if sampling == 'once':
            batch_indices = np.random.choice(self.n, batch_size, replace=False) # n_full or self.n
            A_batch = A_full[batch_indices,::]
            b_batch = b_full[batch_indices]

        
        s_global = 0

        for i in range(n_iter):
            print("Pass "+str(i))
            start = time()

            g0 = self.grad(x)

            # print(g0[:10])

            if stop_averaging == 0 or i < stop_averaging:
                hsqrt = self.sqrt_hess(x)
                sa = SKETCH_FN[sketch](hsqrt * self.A, sketch_size, nnz=nnz)
                if scheme == 'unweighted':
                    w_old = self.uniform_weight(i)
                    w = self.uniform_weight(i + 1)
                    hs_ = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
                    hs = (w_old / w) * hs + (1 - w_old / w) * hs_
                else:
                    hs = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
                u = torch.linalg.cholesky(hs)

            if s_global < 1:
                v = -torch.cholesky_solve(g0, u)
                s_global = self.line_search(x, v, g0)
                x = x + s_global*v
                # print(x[:5].T)
            else:
                x0 = x.clone().detach()
                if sampling == 'once':
                    self.A = A_batch
                    self.b = b_batch
                    self.n = batch_size
                elif sampling == 'per stage':
                    batch_indices = np.random.choice(n_full, batch_size, replace=False)
                    self.A = A_full[batch_indices,::]
                    self.b = b_full[batch_indices]
                    self.n = batch_size
                for j in range(n_local):
                    if sampling == 'per step':
                        batch_indices = np.random.choice(n_full, batch_size, replace=False) # n_full or self.n
                        ghat = self.grad(x,indices=batch_indices)
                        ghat0 = self.grad(x0,indices=batch_indices)
                    else:
                        ghat = self.grad(x)
                        ghat0 = self.grad(x0)
                    if with_vr:
                        g = ghat - ghat0 + g0
                    else:
                        g = ghat
                    v = -torch.cholesky_solve(g, u)
                    x = x + s*v
                    
                self.A = A_full
                self.b = b_full
                self.n = n_full
                v = x - x0
                s_global = self.line_search(x0, v, g0)
                x = x0 + s_global*v
                if not with_vr or permanent_switch: s_global = 1

            times.append(time()-start)
            losses.append(self.loss(x).cpu().numpy().item())
            
            if np.sum(times) > self.MAX_TIME or losses[-1] < self.MIN_LOSS:
                losses = np.append(losses, np.zeros(n_iter - (i + 1)) + losses[-1])
                times = np.append(times, np.zeros(n_iter - (i + 1)))
                break
        
        losses = np.array(losses)
        times = np.array(times)
        losses /= losses[0]
        
        return x, losses, np.cumsum(times)
    
    ########################################
    # First for loop for computing gradient
    ########################################
    def cal_local_grad(self, k, i, sketch_size, sketch='rrs', scheme='no averaging', stop_averaging=0):
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)
        # self.n = int(self.n/self.m)
        # n_full = self.n
        # if n_local == 0:
        #     n_local = max(int(np.log(n_full/self.d)/np.log(2)),2)
        g0_new = torch.zeros(self.m, self.d, self.c).to(self.device)
        H_new = torch.zeros(self.m, self.d, self.d).to(self.device)
        L_new = torch.zeros(self.m, self.d, self.d).to(self.device)
        # for k in range(self.m):
        g0_loc = self.dgrad(x, k)
        # construct local data
        loc_A = self.local_A[k, :, :]
        loc_b = self.local_b[k, :, :]
        if stop_averaging == 0 or i < stop_averaging:
            hsqrt = self.sqrt_hess(x, 1)
            sa = SKETCH_FN[sketch](hsqrt * loc_A, sketch_size, nnz=.1)
        if scheme == 'unweighted':
            w_old = self.uniform_weight(i)
            w = self.uniform_weight(i + 1)
            hs_ = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
            H_new[k] = (w_old / w) * H_new[k] + (1 - w_old / w) * hs_
        else:
            H_new[k] = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)

        loc_decompose_H = torch.linalg.cholesky(H_new[k])
        L_new[k] = loc_decompose_H
        g0_new[k] = g0_loc
        return g0_new[k]
    
    ########################################
    # Second for loop for computing direction
    ########################################
    
    def cal_local_direction(self, k, g0, L_new, n_full, n_local = 0, sampling='per stage', with_vr=True):
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)
        x0 = x.clone().detach()
        if n_local == 0:
            n_local = max(int(np.log(self.n/self.d)/np.log(2)),2)
        batch_size = self.n / n_local # change
        x_new = x0.T.repeat(self.m, 1).to(self.device)

        for j in range(n_local):
            if sampling == 'per step':
                batch_indices = np.random.choice(n_full, int(batch_size), replace=False) # n_full or self.n
                ghat = self.dgrad(x, k, indices=batch_indices)
                ghat0 = self.dgrad(x0, k, indices=batch_indices)
            else:
                ghat = self.dgrad(x, k)
                ghat0 = self.dgrad(x0, k)
            if with_vr:
                g = ghat - ghat0 + g0
            else:
                g = ghat
            v = -torch.cholesky_solve(g, L_new[k])
            x_new[k] += 1*v.T.squeeze(0)
        return x_new[k]

    ########################################################################################################################
    def ihs_dsvrn(self, sketch_size, sketch='rrs', nnz=.1, n_local = 0, n_iter=10, scheme='unweighted', sampling='per stage',with_vr=True,s=1,stop_averaging=0,permanent_switch=False):
        # initial point (just random)
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)
        # hs = torch.zeros(self.d,self.d).to(self.device)

        # self.n = int(self.n/self.m)
        ########################################
        # make sure the local data size all equal to n/m!!!
        ########################################
        
        A_full = self.A
        b_full = self.b
        n_full = self.n
        local_n_full = int(self.n / self.m)

        local_A_full = self.local_A
        local_b_full = self.local_b
        
        ########################################
        losses = [self.loss(x).cpu().numpy().item()]
        times = [0.]
        if n_local == 0:
            n_local = max(int(np.log((self.n)/self.d)/np.log(2)),2)

        ########################################
        # we consider local batch size here!!!!!
        # batch_size = int((self.n / self.m) / n_local) # change
        batch_size = int(local_n_full / n_local) # change
        print("DSVRN with local batch_size=" + str(batch_size) + " and n_local=" + str(n_local)+"\n")
        ########################################
        # if sampling == 'once':
        #     batch_indices = np.random.choice(self.n, batch_size, replace=False) # n_full or self.n
        #     A_batch = A_full[batch_indices,::]
        #     b_batch = b_full[batch_indices]

        ########################################
        # sampling once
        # create batch indeices for each local machine?
        # batch_size = local batch size
        if sampling == 'once':
          # all_A_batch = torch.zeros(m, batch_size, self.d).to(self.device)
          # all_b_batch = torch.zeros(m, batch_size).to(self.device)
          
          batch_indices = np.random.choice(self.local_data_size, (self.m, batch_size), replace=True) # n_full or self.n #change
          # batch_indices = np.random.choice(n_full, (self.m, batch_size), replace=False) # change
          all_A_batch = local_A_full[torch.arange(self.m).unsqueeze(1), batch_indices, :]
          all_b_batch = local_b_full[torch.arange(self.m).unsqueeze(1), batch_indices, :]
        ########################################

        
        s_global = 0
        ########################################
        # create two lists to store local gradients and Hessian
        # this is 2-D with shape:  m*d
        g0_new = torch.zeros(self.m, self.d, self.c).to(self.device)
        # this is 3-D with shape: m*d*d
        H_new = torch.zeros(self.m, self.d, self.d).to(self.device)
        # this is 3-D with shape: m*d*d
        L_new = torch.zeros(self.m, self.d, self.d).to(self.device)
        ########################################

        for i in range(n_iter):
            print("Pass "+str(i))
            start = time()
            ########################################
            # compute local grad
            # parallel part !!!!!!!!!!!!!!!!!!!!!!!!
            
            for k in range(self.m):
                ########################################
                # compute local gradient
                g0_loc = self.dgrad(x, k)
                
                # loc_b = self.local_b[k, :, :]
                ########################################
                # compute local Hessian via sub-sampling sketch for each machine
                if stop_averaging == 0 or i < stop_averaging:
                  hsqrt = 1./np.sqrt(self.local_data_size)
                  sa = SKETCH_FN[sketch](hsqrt * self.local_A[k, :, :], int(sketch_size / self.m), nnz=nnz/self.m)
                  if scheme == 'unweighted':
                      w_old = self.uniform_weight(i)
                      w = self.uniform_weight(i + 1)
                      hs_ = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
                      H_new[k] = (w_old / w) * H_new[k] + (1 - w_old / w) * hs_
                  else:
                      H_new[k] = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)

                    
                
                  # store local Hessian('s decomposition) into H
                  L_new[k] = torch.linalg.cholesky(H_new[k])
                  # store local gradient in g0_new
                  g0_new[k] = g0_loc

            g0 = torch.sum(g0_new, dim=0)
                ########################################

           
            ########################################
            # Calculate for loop by parallelize
            ########################################
             # #     # store local gradient into g0
            
                # H_loc = self.hessian(x, loc_A, loc_b)
                # u_loc = torch.linalg.cholesky(H_loc)
                # H_new[i] = H
            # g0_new, L_new = self.cal_local_grad(i, sketch_size, sketch='rrs', scheme='no averaging', stop_averaging=0)
            # with parallel_backend('threading', n_jobs=self.m):
                # res = Parallel()(delayed(self.cal_local_grad)(k, i, sketch_size, sketch='rrs', scheme='no averaging', stop_averaging=0) for k in range(self.m))
            #     # g0_new = torch.FloatTensor(g0_new)
                # g0_new = [item[0] for item in res]
            #     # L1_new  = [item[1] for item in res]
                # print(res[0][0] == res[1][0])
                # print(res[0][1] == res[1][1])
                # print(g0_new)
                # g0_new = torch.tensor(g0_new)
                # L_new = torch.tensor(L_new)
            # g0_new = Parallel(n_jobs=self.m)(delayed(self.cal_local_grad(a)) for a in range(self.m))
            # print(len(g0_new[0]))
            # # print(len(g0_new[0]))
            # print(len(g0_new[0][0]))
            # print(len(g0_new[0][0][0]))
            
            # print(g0_new[0].shape)
            # print(g0_new[1].shape)
            
            ########################################
            # perform communication round here
            # sum up all the local gradient to compute full gradient
            # g0 = torch.zeros(self.d, self.c).to(self.device)
            
            # for i in range(self.m):
            #     g0 += g0_new[i]
            # print(aa.shape)
            # print(res[0][0].shape)
            # print(res[0][1].shape)
            # print(len(res[0]))
            # print(g0.shape)
            ########################################
            
            if s_global < 1:

                ########################################
                # perform communication round here
                v = torch.zeros(self.d, self.c).to(self.device)
                # if step size < 1, use ordinary solve
                for k in range(self.m):
                    v += -torch.cholesky_solve(g0, L_new[k]) / self.m
                s_global = self.line_search(x, v, g0)
                # update x
                x = x + s_global * v
                # print(x[:5].T)
                ########################################
            else:
                x0 = x.clone().detach()
                if sampling == 'once':
                    self.local_A = all_A_batch
                    self.local_b = all_b_batch
                    self.n = batch_size
                elif sampling == 'per stage':
                    
                    batch_indices = np.random.choice(self.local_data_size, (self.m, int(batch_size)), replace=True) # n_full or self.n
                    batch_indices = torch.tensor(batch_indices)
                    
                    self.local_A = local_A_full[torch.arange(self.m).unsqueeze(1), batch_indices, :]
                    self.local_b = local_b_full[torch.arange(self.m).unsqueeze(1), batch_indices, :]
                    self.local_data_size = batch_size
                # print('A1:', self.local_A.shape)
                # print('b1:',self.local_b.shape)
                ########################################
                
                x_new = x0.T.repeat(self.m, 1).to(self.device)
                # for each local machine
                for k in range(self.m):
                    
                    # for each local step
                    for j in range(n_local):
                        # v_new = []
                        if sampling == 'per step':
                            batch_indices = np.random.choice(self.n, int(batch_size), replace=True) # n_full or self.n
                            ghat = self.dgrad(x_new[k].unsqueeze(1), k, indices=batch_indices)
                            ghat0 = self.dgrad(x0, k, indices=batch_indices)
                        else:
                            # print(self.local_A.shape[1])
                            # print(self.local_b.shape[1])
                            ghat = self.dgrad(x_new[k].unsqueeze(1), k)
                            ghat0 = self.dgrad(x0, k)
                        if with_vr:
                            g = ghat - ghat0 + g0
                        else:
                            g = ghat
                        v = -torch.cholesky_solve(g, L_new[k])
                        x_new[k] += s*v.T.squeeze(0)
                        
                      ########################################
                
                ########################################
                # Using joblib to parallel calculation
                # with parallel_backend('threading', n_jobs=self.m):
                #     x_new = Parallel()(delayed(self.cal_local_direction)(k, g0, L_new, n_full, n_local = 0, sampling='per stage', with_vr=True) for k in range(self.m))
                    
                ########################################
                # perform communication round here
                # average over all the returne
                v_sum = (1/self.m) * torch.sum(x_new, dim=0,keepdim=True).T - x0
                s_global = self.line_search(x0, v_sum, g0)
                x = x0 + s_global*v_sum
                ########################################
                self.A = A_full
                self.b = b_full
                ########################################
                # all_A_local = torch.zeros(self.m, self.n/self.m, self.d).to(self.device)
                # all_b_local = torch.zeros(self.m, self.n/self.m).to(self.device)
                # for i in range(self.m):
                #   all_A_local[i] = self.A[(self.n / self.m) * i : (self.n / self.m) * (1 + i) - 1, :]
                #   all_b_local[i] = self.b[(self.n / self.m) * i : (self.n / self.m) * (1 + i) - 1]
                self.local_A = local_A_full
                self.local_b = local_b_full
                self.local_data_size = local_n_full
                self.n = n_full
                # v = x - x0
                
                if not with_vr or permanent_switch: s_global = 1

            times.append(time()-start)
            losses.append(self.loss(x).cpu().numpy().item())
            
            if np.sum(times) > self.MAX_TIME or losses[-1] < self.MIN_LOSS:
                losses = np.append(losses, np.zeros(n_iter - (i + 1)) + losses[-1])
                times = np.append(times, np.zeros(n_iter - (i + 1)))
                break
        
        losses = np.array(losses)
        times = np.array(times)
        losses /= losses[0]

        
        return x, losses, np.cumsum(times)
      ########################################################################################################################

    def svrg(self, m, n_iter=100, s=0.01, batch_size=10):
        
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)

        losses = [self.loss(x).cpu().numpy().item()]
        times = [0.]
        print("\nSVRG with s="+str(s)+" m="+str(m)+" b="+str(batch_size))
        for i in range(n_iter):
            print("Pass "+str(i))
            start = time()

            g = self.grad(x)
            x0 = x.clone().detach()

            for j in range(m):
                batch_indices = np.random.choice(n_full, batch_size, replace=False)
                
                g_sto = self.grad(x, indices = batch_indices)
                g_sto0 = self.grad(x0, indices = batch_indices)
                x = x - s*(g_sto - g_sto0 + g)
            
            times.append(time()-start)
            losses.append(self.loss(x).cpu().numpy().item())
            if np.sum(times) > self.MAX_TIME or losses[-1] < self.MIN_LOSS:
                losses = np.append(losses, np.zeros(n_iter - (i + 1)) + losses[-1])
                times = np.append(times, np.zeros(n_iter - (i + 1)))
                break

            
        losses = np.array(losses)
        times = np.array(times)
        losses /= losses[0]
            
        return x, losses, np.cumsum(times)
    
    
