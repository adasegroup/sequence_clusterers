import numpy as np
from utils.metrics import purity
import torch

class Trainer_single:
    def __init__(self, model, optimizer, criterion, X, val,\
                 max_epochs = 100, batch_size = 30, generator_model = None):
        self.N = X.shape[0]
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.X = X
        self.val = val
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.generator_model = generator_model
    def train_epoch(self, epoch):
        indices = np.random.permutation(self.N)
        self.model.train()
        log_likelihood = []
        val_ll = None
        mse = []
        val_mse = None
        for iteration, start in enumerate(range(0, self.N - self.batch_size, self.batch_size)):
            batch_ids = indices[start:start+self.batch_size]
            batch = self.X[batch_ids]
            self.optimizer.zero_grad()
            lambdas = self.model(batch)
            if self.generator_model:
                true_lambdas = self.generator_model(batch)
            loss = self.criterion(batch, lambdas, batch[:,0,0])
            loss.backward()
            self.optimizer.step()
            log_likelihood.append(loss.item())
            if self.generator_model:
                mse.append(np.var((lambdas.detach().numpy() - true_lambdas.detach().numpy())))
        self.model.eval()
        lambdas = self.model(self.val)
        if self.generator_model:
            true_lambdas = self.generator_model(self.val)
        val_ll = self.criterion(self.val, lambdas, self.val[:,0,0])
        if self.generator_model:
            val_mse = np.var((lambdas.detach().numpy() - true_lambdas.detach().numpy()))
        return log_likelihood, mse, val_ll, val_mse
    def train(self):
        self.generator_model.eval()
        losses = []
        val_losses = []
        mses = []
        val_mses = []
        for epoch in range(self.max_epochs):
            ll, mse, val_ll, val_mse =  self.train_epoch(epoch)
            losses.append(np.mean(ll))
            val_losses.append(val_ll)
            mses.append(np.mean(mse))
            val_mses.append(val_mse)
            if len(mse):
                print('On epoch {}/{}, ll = {}, mse = {}, val_ll = {}, val_mse = {}'\
                      .format(epoch, self.max_epochs,\
                              np.mean(ll), np.mean(mse), val_ll, val_mse))
            else:
                print('On epoch {}/{}, ll = {}, val_ll = {}'.format(epoch, self.max_epochs,\
                                                                np.mean(ll), val_ll))
        return losses, val_losses, mses, val_mses
    
class Trainer_clusterwise:
    def __init__(self, model, optimizer, device, X_train, X_val, target_train, target_val, n_clusters,\
                 alpha = 1.002, beta = 0.001, epsilon = 1e-3, l = 0.5, eps = 0.1, max_epochs = 100, max_m_step_epochs = 30, lr_update_tol = 15, lr_update_param = 0.1, batch_size = 100):
        self.N = X_train.shape[0]
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.X_train = X_train.to(device) #[N, seq_len, n_classes + 1]
        self.X_val = X_val.to(device)
        self.target_train = target_train.to(device) #[N]
        self.target_val = target_val.to(device)
        self.n_clusters = n_clusters
        self.max_epochs = max_epochs
        self.lr_update_tol = lr_update_tol
        self.update_checker = -1
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.l = l
        self.eps = eps
        self.lr_update_param = lr_update_param
        self.prev_loss = 0
        self.max_m_step_epochs = max_m_step_epochs
        self.batch_size = batch_size
        self.pi = (torch.ones(n_clusters)/n_clusters).to(device) #[n_classes]
        self.gamma = torch.zeros(n_clusters,X_train.shape[0]).to(device) #[n_classes, N]
    def test_sub_loss(self, l, name):
        l.backward(retain_graph = True)
        for p in self.model.parameters():
            if p.grad.sum()!=p.grad.sum():
                print(name,'value:',l, 'Fails')
                self.optimizer.zero_grad()
                return False
        self.optimizer.zero_grad()
        return True
    def loss(self, partitions, lambdas, gamma, test_losses = False):
        new_gamma = self.compute_gamma(lambdas, partitions, (self.n_clusters, partitions.shape[0]), self.device).to(self.device)
#         new_gamma[new_gamma>=(1-self.eps)] = 1
#         new_gamma[new_gamma<self.eps] = 0
        h = self.l*torch.sum(torch.sum(new_gamma, dim = 1)*torch.log(torch.sum(new_gamma, dim = 1)/partitions.shape[0]+self.epsilon))
        dts = partitions[:,0,0].to(self.device)
        dts = dts[None,:,None,None].to(self.device)
        tmp = lambdas*dts
        p = partitions[None,:,:,1:].to(self.device)
        tmp1 = tmp - p*torch.log(tmp+self.epsilon) + torch.lgamma(p+1)
        tmp2 = torch.sum(tmp1, dim = (2,3))
        gamma[gamma<self.eps] = 0
        gamma[gamma>=(1-self.eps)] = 1
        tmp3 = gamma*tmp2
        if torch.sum(tmp3==tmp3)==0:
            print('Everything failed!')
            return torch.sum(tmp3)
        loss1 = torch.sum(tmp3)
        loss2 = - torch.sum((self.alpha-1)*torch.log(lambdas+self.epsilon) - self.beta*lambdas**2)
        ent = True
        if test_losses:
            self.test_sub_loss(loss1,'Main')
            self.test_sub_loss(loss2,'Regularization')
            ent = self.test_sub_loss(h, 'Entropy')
        if ent:
            res = loss1 + h + loss2
        else:
            res = loss1 + loss2
        return res
    def compute_gamma(self, lambdas, X = None, size = None, device = 'cpu'):
        if size == None:
            gamma = torch.zeros_like(self.gamma)
        else:
            gamma = torch.zeros(size)
        if X == None:
            dts = self.X_train[:,0,0].to(device)
            dts = dts[None,:,None,None].to(device)
            partitions = self.X_train[:,:,1:].to(device)
            partitions = partitions[None,:,:,:].to(device)
        else:
            dts = X[:,0,0].to(device)
            dts = dts[None,:,None,None].to(device)
            partitions = X[:,:,1:].to(device)
            partitions = partitions[None,:,:,:].to(device)
        for k in range(self.n_clusters):
            lambdas_k = lambdas[k,:,:,:]
            lambdas_k = lambdas_k[None,:,:,:]
            w = self.pi/self.pi[k]
            w = w[:,None].to(device)
            tmp_sub = (lambdas.to(device) - lambdas_k.to(device))*dts.to(device)
            #tmp_div = lambdas.to(device)/(lambdas_k.to(device) + self.epsilon)
            tmp = torch.sum( - tmp_sub + partitions*(torch.log(lambdas.to(device) + self.epsilon) - torch.log(lambdas_k.to(device) + self.epsilon)), dim = (2,3))
            tmp = 1/(torch.sum(w * torch.exp(tmp), dim = 0))
            tmp[tmp!=tmp] = 0
            gamma[k,:] = tmp
#         for n in range(partitions.shape[1]):
#             N = partitions.shape[1]
#             if torch.sum(gamma[:,n])!=1:
#                 gamma[:,n] += (1 - torch.sum(gamma[:,n]))/self.n_clusters
        return gamma
    def e_step(self):
        self.model.eval()
        with torch.no_grad():
            lambdas = self.model(self.X_train)
            self.gamma = self.compute_gamma(lambdas)
            print(torch.sum(self.gamma))
    def train_epoch(self, epoch):
        indices = np.random.permutation(self.N)
        self.model.train()
        log_likelihood = []
        checker = 0
        for iteration, start in enumerate(range(0, self.N - self.batch_size, self.batch_size)):
            batch_ids = indices[start:start+self.batch_size]
            batch = self.X_train[batch_ids].to(self.device)
            self.optimizer.zero_grad()
            lambdas = self.model(batch).to(self.device)
            loss = self.loss(batch, lambdas, self.gamma[:,batch_ids], True)
            loss.backward()
#             bad = False
#             for param in self.model.parameters():
#                 tmp = torch.sum(param.grad)
#                 if tmp!=tmp:
#                     bad = True
#             if bad:
#                 print('Bad batch')
#                 continue
#             if checker == 0:
#                 for param in self.model.parameters():
#                     print('Param:',param, 
#                           '\nGrads:',param.grad)
#                     if param.grad.sum() != param.grad.sum():
#                         print(loss)
#                         checker = 1
#                         break
            self.optimizer.step()
            log_likelihood.append(loss.item())
        if np.mean(log_likelihood) > self.prev_loss:
            self.update_checker += 1
            if self.update_checker >= self.lr_update_tol:
                self.update_checker = 0
                print('Updating lr')
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.lr_update_param                        
        self.prev_loss = np.mean(log_likelihood)
        return log_likelihood
    def m_step(self):
        #self.pi = torch.sum(self.gamma, dim = 1)/self.N
        log_likelihood_curve = []
        for epoch in range(self.max_m_step_epochs):
            ll = self.train_epoch(epoch)
            log_likelihood_curve += ll
            if np.mean(ll) != np.mean(ll):
                return None, None, None
            if epoch%10 == 0:
                print('Loss on sub_epoch {}/{}: {}'.format(epoch+1,\
                                                            self.max_m_step_epochs,\
                                                            np.mean(ll)))
        self.model.eval()
        with torch.no_grad():
            lambdas = self.model(self.X_train)
            gamma = self.compute_gamma(lambdas)
            clusters = torch.argmax(gamma, dim = 0)
            print('Cluster partition')
            tmp = 2
            for i in np.unique(clusters.cpu()):
                print('Cluster',i,': ',np.sum((clusters.cpu() == i).cpu().numpy())/len(clusters),' with pi = ',self.pi[i])
                tmp = min(tmp, np.sum((clusters.cpu() == i).cpu().numpy())/len(clusters))
            pur = purity(clusters,self.target_train)
        return log_likelihood_curve, [np.mean(ll), pur], tmp
    def train(self):
        losses = []
        purities = []
        purities_val = []
        for epoch in range(self.max_epochs):
            #if epoch!=0:
            #    torch.save(self.model.state_dict(), 'reserv.pt')
            print('Beginning e-step')
            self.e_step()
            if epoch == 0:
                clusters = torch.argmax(self.gamma, dim = 0)
                print('Cluster partition')
                for i in np.unique(clusters.cpu()):
                    print('Cluster',i,': ',np.sum((clusters.cpu() == i).cpu().numpy())/len(clusters),' with pi = ',self.pi[i])
                random_pur = purity(clusters,self.target_train)
                print('Purity for random model: {}'.format(random_pur))
            print('Beginning m-step')
            ll, ll_pur, cluster_part =  self.m_step()
            if ll == None:
                return None, None, None, None
            losses+=ll
            purities.append(ll_pur + [cluster_part])
            print('On epoch {}/{} average loss = {}, purity = {}'.format(epoch+1,\
                                                                         self.max_epochs,\
                                                                         np.mean(ll), ll_pur[1]))
            self.model.eval()
            with torch.no_grad():
                lambdas = self.model(self.X_val).to(self.device)
                gamma = self.compute_gamma(lambdas, self.X_val, (self.n_clusters, self.X_val.shape[0]))
                ll = self.loss(self.X_val.to(self.device),lambdas.to(self.device), gamma.to(self.device))
                clusters = torch.argmax(gamma, dim = 0)
                pur = purity(clusters.cpu(),self.target_val.cpu())
                purities_val.append([ll.item(), pur])
                print('Validation loss = {}, purity = {}'.format(ll.item(),pur))
        return losses, purities, purities_val, cluster_part
    
    
    
    
    
    
    
    
    
    
    
    