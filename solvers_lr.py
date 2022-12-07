import numpy as np
import torch
from time import time

from sketches import gaussian, less, sparse_rademacher, srht, rrs, rrs_lev_scores

SKETCH_FN = {'gaussian': gaussian, 'less': less, 'less_sparse': sparse_rademacher, 
             'srht': srht, 'rrs': rrs, 'rrs_lev_scores': rrs_lev_scores}


torch.set_default_dtype(torch.float64)

class LogisticRegression:
    
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
        self.ls_passes = 1
        #############################
        self.m = machine
        self.local_A = self.A.view(self.m, int(self.n / self.m), -1)
        self.local_b = self.b.view(self.m, int(self.n / self.m), -1)
        self.local_data_size = int(self.n / self.m)
        
        
    def loss(self, x):
        return 1./2 * ((self.H_opt @ (x - self.x_opt))**2).sum()
    
    def logistic_loss(self, x):
        return (torch.log(1 + torch.exp(-self.b * (self.A @ x)))).mean() + self.lambd/2 * (x**2).sum()

    def grad(self, x, indices=[]):
        batch_size = len(indices)
        if batch_size>1:
            return -1./batch_size*self.A[indices,::].T @ ( self.b[indices] * 1./(1+torch.exp(self.b[indices] * (self.A[indices,::] @ x))))+ self.lambd * x
        elif batch_size==0:
            return -1./self.n*self.A.T @ ( self.b * 1./(1+torch.exp(self.b * (self.A @ x))))+ self.lambd * x
        else:
            index = indices[0]
            a_i = self.A[index,::].reshape((-1,1))
            b_i = self.b[index].squeeze()
            return -a_i.reshape((-1,1)) * (b_i /(1+torch.exp(b_i * ( (a_i*x).sum()))))+ self.lambd * x
    
    def dgrad(self, x, m_num, indices=[]):
        batch_size = len(indices)
        local_data_size = self.local_data_size
        if self.local_A[m_num].shape[0] != local_data_size or self.local_b[m_num].shape[0] != local_data_size:
              print("wrong local data size!!")
        if batch_size>1:
            return 1./batch_size * self.local_A[m_num][indices,::].T \
             @ (self.b[m_num][indices] * 1./ (1 + torch.exp(self.local_b[m_num][indices] * (self.local_A[m_num][indices,::] @ x)))) + self.lambd * x
        elif batch_size==0:
            return -1./local_data_size * self.local_A[m_num].T @ ( self.local_b[m_num] * 1./(1+torch.exp(self.local_b[m_num] * (self.local_A[m_num] @ x))))+ self.lambd * x
        else:
            index = indices[0]
            # index = np.random.choice(self.n)
            a_i = self.local_A[m_num][index,::].reshape((-1,1))
            b_i = self.local_b[m_num][index].squeeze()
            return -a_i.reshape((-1,1)) * (b_i /(1+torch.exp(b_i * ( (a_i*x).sum()))))+ self.lambd * x
    
    def hessian(self, x):
        Ax = self.A @ x
        v = torch.exp(self.b * Ax)
        D = v / (1+v)**2
        return 1./self.n * self.A.T @ (D * self.A) + self.lambd * torch.eye(self.d).to(self.device)

    def newton(self, n_iter=10):
        
        x = 1./np.sqrt(self.d)*torch.randn(self.d, self.c).to(self.device)

        print("Newton\n")
        losses = [self.loss(x).cpu().numpy().item()]
        times = [0.]
        for _ in range(n_iter):
            
            start = time()
            g = self.grad(x)
            H = self.hessian(x)
            u = torch.linalg.cholesky(H)
            v = -torch.cholesky_solve(g, u)
            delta = - (g * v).sum()
            s = self.line_search(x, v, g)
            x = x + s*v
            times.append(time()-start)
            losses.append(self.loss(x).cpu().numpy().item())

            if np.sum(times) > self.MAX_TIME or losses[-1] < self.MIN_LOSS:
                losses = np.append(losses, np.zeros(n_iter - (_ + 1)) + losses[-1])
                times = np.append(times, np.zeros(n_iter - (_ + 1)))
                break
        
        losses = np.array(losses)
        times = np.array(times)
        losses /= losses[0]

        return x, losses, np.cumsum(times)

    def uniform_weight(self, iter_num):
        return iter_num 
    
    def poly_weight(self, iter_num, p=2):
        return (iter_num) ** p

    def log_weight(self, iter_num):
        return (iter_num) ** np.log(iter_num+2)

    def weight(self, iter_num, scheme="unweighted"):
        if scheme == "poly":
            return self.poly_weight(iter_num)
        elif scheme == "log":
            return self.log_weight(iter_num)
        else:
            return self.uniform_weight(iter_num)

    def sqrt_hess(self, x):
        v_ = torch.exp(self.b * (self.A @ x))
        return 1./np.sqrt(self.n)*torch.sqrt(v_) / (1+v_)

    def line_search(self, x, v, g, alpha=0.3, beta=0.8):
        delta = (v*g).sum()
        loss_x = self.logistic_loss(x)
        s = 1
        xs = x + s*v
        self.ls_passes = 1
        while self.logistic_loss(xs) > loss_x + alpha*s*delta:
            s = beta*s 
            xs = x + s*v
            self.ls_passes += 1
        print("line search: passes = " + str(self.ls_passes) + ", step size = " + str(s))        
        return s
    
    def solve_exactly(self, n_iter=100, eps=1e-8):
        losses = []
        x = 1./np.sqrt(self.d)*torch.randn(self.d, self.c).to(self.device)
        
        for _ in range(n_iter):
            losses.append(self.logistic_loss(x).cpu().numpy().item())
            g = self.grad(x)
            H = self.hessian(x)
            u = torch.linalg.cholesky(H)
            v = -torch.cholesky_solve(g, u)
            # v = -torch.pinverse(H) @ g
            delta = - (g * v).sum()
            if delta < eps:
                break
            s = self.line_search(x, v, g)
            x = x + s*v
        
        self.x_opt = x
        self.H_opt = H
        
        _, sigma, _ = torch.svd(self.H_opt)
        
        de_ = torch.trace((H - self.lambd * torch.eye(self.d).to(self.device)) @ torch.pinverse(H))
        self.de = de_.cpu().numpy().item()
        
        return x, losses
    
    def ihs_no_averaging(self, x, sketch_size, sketch, nnz):
        
        start = time()
        
        hsqrt = self.sqrt_hess(x).reshape((-1,1))
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
        
        hsqrt = self.sqrt_hess(x).reshape((-1,1))
        sa = SKETCH_FN[sketch](hsqrt * self.A, sketch_size, nnz=nnz)
        g = self.grad(x)

        w_old = self.uniform_weight(iter)
        w = self.uniform_weight(iter + 1)
        hs_ = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
        hs = (w_old / w) * hs_old + (1 - w_old / w) * hs_
            
        u = torch.linalg.cholesky(hs)
        v = -torch.cholesky_solve(g, u)
        # v = -torch.pinverse(hs) @ g
        
        s = self.line_search(x, v, g)
        x = x + s*v
                
        return x, hs, time()-start

    
    def ihs(self, sketch_size, sketch='gaussian', nnz=1., n_iter=10, scheme='unweighted'):
        
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)
        hs = torch.zeros(self.d,self.d).to(self.device)

        print("IHS "+scheme + "\n") 
        losses = [self.loss(x).cpu().numpy().item()]
        times = [0.]
        for i in range(n_iter):
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


    def ihs_svrn(self, sketch_size, sketch='rrs', nnz=.1, n_local = 0, n_iter=10, scheme='no averaging', sampling='per stage',with_vr=True,s=1):
                
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
            batch_indices = np.random.choice(self.n, batch_size, replace=False)
            A_batch = A_full[batch_indices,::]
            b_batch = b_full[batch_indices]

        s_global = 0
        
        for i in range(n_iter):
            start = time()

            g0 = self.grad(x)
            
            hsqrt = self.sqrt_hess(x).reshape((-1,1))
            sa = SKETCH_FN[sketch](hsqrt * self.A, sketch_size, nnz=nnz)
            if scheme == 'no averaging':
                hs = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
            else:
                w_old = self.weight(i,scheme=scheme)
                w = self.weight(i+1,scheme=scheme)
                hs_ = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
                hs = (w_old / w) * hs + (1 - w_old / w) * hs_
            u = torch.linalg.cholesky(hs)          

            if s_global < 1:
                v = -torch.cholesky_solve(g0, u)
                s_global = self.line_search(x, v, g0)
                x = x + s_global*v
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
                        batch_indices = np.random.choice(n_full, batch_size, replace=False)
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

    def ihs_dsvrn(self, sketch_size, sketch='rrs', nnz=.1, n_local = 0, n_iter=10, scheme='no averaging', sampling='per stage', with_vr=True, s=1):
        # initial point (just random)
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)

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
        elapsed_time = [0.]
        if n_local == 0:
            n_local = max(int(np.log((self.n)/self.d)/np.log(2)),2)

        ########################################
        # we consider local batch size here!!!!!
        batch_size = int((self.n / self.m) / n_local) # change
        # batch_size = int(local_n_full / n_local) # change
        print("DSVRN with local batch_size=" + str(batch_size) + " and n_local=" + str(n_local)+"\n")
        ########################################
        
        if sampling == 'once':    
          batch_indices = np.random.choice(self.local_data_size, (self.m, batch_size), replace=True) # n_full or self.n #change
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
                # print(g0_loc)
                # loc_b = self.local_b[k, :, :]
                ########################################
                # compute local Hessian via sub-sampling sketch for each machine
                hsqrt = 1./np.sqrt(self.local_data_size)
                sa = SKETCH_FN[sketch](hsqrt * self.local_A[k, :, :], int(sketch_size / self.m), nnz=nnz/self.m)
                if scheme == 'no averaging':
                    H_new[k] = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
                else:
                    w_old = self.weight(i,scheme=scheme)
                    w = self.weight(i+1,scheme=scheme)
                    hs_ = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
                    H_new[k] = (w_old / w) * H_new[k] + (1 - w_old / w) * hs_
                
                # store local Hessian('s decomposition) into H
                L_new[k] = torch.linalg.cholesky(H_new[k])
                # store local gradient in g0_new
                g0_new[k] = g0_loc
            
                ########################################
            elapsed_time.append(time()-start)
            # print(elapsed_time_lc)
            ########################################
            # Calculate for loop by parallelize
            ########################################
            # Parallel(n_jobs=self.m, backend="threading")(delayed(self.cal_local_grad)(k, i, g0_new, L_new, H_new, sketch_size, sketch, scheme, stop_averaging, nnz) for k in range(self.m))          
            g0 = torch.sum(g0_new, dim=0)
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
                print("Hello")
                x0 = x.clone().detach()
                if sampling == 'once':
                    self.local_A = all_A_batch
                    self.local_b = all_b_batch
                    self.n = batch_size
                elif sampling == 'per stage':
                    
                    batch_indices = np.random.choice(self.local_data_size, (self.m, int(batch_size)), replace=False) # n_full or self.n
                    batch_indices = torch.tensor(batch_indices)
                    batch_indices = batch_indices.type(torch.long)
                    
                    self.local_A = local_A_full[torch.arange(self.m).unsqueeze(1), batch_indices, :]
                    self.local_b = local_b_full[torch.arange(self.m).unsqueeze(1), batch_indices, :]
                    self.local_data_size = batch_size
                ########################################
                
                x_new = x0.T.repeat(self.m, 1).to(self.device)
                # for each local machine
                for k in range(self.m):
                    
                    # for each local step
                    for j in range(n_local):
                        # v_new = []
                        if sampling == 'per step':
                            batch_indices = np.random.choice(self.n, int(batch_size), replace=False) # n_full or self.n
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
                ########################################
                # Parallel(n_jobs=self.m, backend="threading")(delayed(self.cal_local_direction)(k, g0, L_new, batch_size, x_new, x0, n_local, sampling, with_vr) for k in range(self.m))
                    
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
                self.local_A = local_A_full
                self.local_b = local_b_full
                self.local_data_size = local_n_full
                self.n = n_full
                

            times.append(time()-start)
            losses.append(self.loss(x).cpu().numpy().item())
            
            if np.sum(times) > self.MAX_TIME or losses[-1] < self.MIN_LOSS:
                losses = np.append(losses, np.zeros(n_iter - (i + 1)) + losses[-1])
                times = np.append(times, np.zeros(n_iter - (i + 1)))
                break
        
        losses = np.array(losses)
        times = np.array(times)
        elapsed_time = np.array(elapsed_time)
        losses /= losses[0]

        
        return x, losses, np.cumsum(times), elapsed_time
    
    def ihs_dsvrn1(self, sketch_size, sketch='rrs', nnz=.1, n_local = 0, n_iter=10, scheme='no averaging', sampling='per stage', with_vr=True, s=1):
        # initial point (just random)
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)

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
        elapsed_time = [0.]
        if n_local == 0:
            n_local = max(int(np.log((self.n)/self.d)/np.log(2)),2)

        ########################################
        # we consider local batch size here!!!!!
        batch_size = int((self.n / self.m) / n_local) # change
        # batch_size = int(local_n_full / n_local) # change
        print("DSVRN with local batch_size=" + str(batch_size) + " and n_local=" + str(n_local)+"\n")
        ########################################
        
        if sampling == 'once':    
          batch_indices = np.random.choice(self.local_data_size, (self.m, batch_size), replace=True) # n_full or self.n #change
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
                # print(g0_loc)
                # loc_b = self.local_b[k, :, :]
                ########################################
                # compute local Hessian via sub-sampling sketch for each machine
                hsqrt = 1./np.sqrt(self.local_data_size)
                sa = SKETCH_FN[sketch](hsqrt * self.local_A[k, :, :], int(self.n / self.m), nnz=nnz/self.m)
                if scheme == 'no averaging':
                    H_new[k] = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
                else:
                    w_old = self.weight(i,scheme=scheme)
                    w = self.weight(i+1,scheme=scheme)
                    hs_ = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
                    H_new[k] = (w_old / w) * H_new[k] + (1 - w_old / w) * hs_
                
                # store local Hessian('s decomposition) into H
                L_new[k] = torch.linalg.cholesky(H_new[k])
                # store local gradient in g0_new
                g0_new[k] = g0_loc
            
                ########################################
            elapsed_time.append(time()-start)
            # print(elapsed_time_lc)
            ########################################
            # Calculate for loop by parallelize
            ########################################
            # Parallel(n_jobs=self.m, backend="threading")(delayed(self.cal_local_grad)(k, i, g0_new, L_new, H_new, sketch_size, sketch, scheme, stop_averaging, nnz) for k in range(self.m))          
            g0 = torch.sum(g0_new, dim=0)
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
                print("Hello")
                x0 = x.clone().detach()
                if sampling == 'once':
                    self.local_A = all_A_batch
                    self.local_b = all_b_batch
                    self.n = batch_size
                elif sampling == 'per stage':
                    
                    batch_indices = np.random.choice(self.local_data_size, (self.m, int(batch_size)), replace=False) # n_full or self.n
                    batch_indices = torch.tensor(batch_indices)
                    batch_indices = batch_indices.type(torch.long)
                    
                    self.local_A = local_A_full[torch.arange(self.m).unsqueeze(1), batch_indices, :]
                    self.local_b = local_b_full[torch.arange(self.m).unsqueeze(1), batch_indices, :]
                    self.local_data_size = batch_size
                ########################################
                
                x_new = x0.T.repeat(self.m, 1).to(self.device)
                # for each local machine
                for k in range(self.m):
                    
                    # for each local step
                    for j in range(n_local):
                        # v_new = []
                        if sampling == 'per step':
                            batch_indices = np.random.choice(self.n, int(batch_size), replace=False) # n_full or self.n
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
                ########################################
                # Parallel(n_jobs=self.m, backend="threading")(delayed(self.cal_local_direction)(k, g0, L_new, batch_size, x_new, x0, n_local, sampling, with_vr) for k in range(self.m))
                    
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
                self.local_A = local_A_full
                self.local_b = local_b_full
                self.local_data_size = local_n_full
                self.n = n_full
                

            times.append(time()-start)
            losses.append(self.loss(x).cpu().numpy().item())
            
            if np.sum(times) > self.MAX_TIME or losses[-1] < self.MIN_LOSS:
                losses = np.append(losses, np.zeros(n_iter - (i + 1)) + losses[-1])
                times = np.append(times, np.zeros(n_iter - (i + 1)))
                break
        
        losses = np.array(losses)
        times = np.array(times)
        elapsed_time = np.array(elapsed_time)
        losses /= losses[0]

        
        return x, losses, np.cumsum(times), elapsed_time
    
    def svrg(self, m, n_iter=100, s=0.01, batch_size=10):
        
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)
        A_full = self.A
        b_full = self.b
        n_full = self.n

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
                #self.A = A_full[batch_indices,::]
                #self.b = b_full[batch_indices]
                #self.n = batch_size
                
                g_sto = self.grad(x, indices = batch_indices)
                g_sto0 = self.grad(x0, indices = batch_indices)
                x = x - s*(g_sto - g_sto0 + g)
            
            self.A = A_full
            self.b = b_full
            self.n = n_full
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
    
