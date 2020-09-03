import torch
import numpy as np
import time
from torch import Tensor
from matplotlib import pyplot as plt
import logging
from gpytorch.mlls import VariationalELBO

import bvcopula
import utils
from . import conf

def plot_loss(filename, losses, rbf, means):
	# prot loss function and kernel length
	fig, (loss, kern, mean) = plt.subplots(1,3,figsize=(16,2))
	loss.plot(losses)
	loss.set_xlabel("Epoch #")
	loss.set_ylabel("Loss")
	marg = (np.max(losses) - np.min(losses))*0.1
	loss.set_ylim(np.min(losses)-marg,
	              np.max(losses)+marg)
	rbf=np.array(rbf).squeeze()
	kern.plot(rbf)
	kern.set_xlabel("Epoch #")
	kern.set_ylabel("Kernel scale parameter")
	mean.plot([np.mean(x,axis=1) for x in means])
	mean.set_xlabel("Epoch #")
	mean.set_ylabel("Mean f")
	fig.savefig(filename)
	plt.close()

def infer(bvcopulas, train_x: Tensor, train_y: Tensor, device: torch.device,
			theta_sharing=None,
			output_loss = None):

	if device!=torch.device('cpu'):
		with torch.cuda.device(device):
			torch.cuda.empty_cache()

	logging.info('Trying {}'.format(utils.get_copula_name_string(bvcopulas)))

	# define the model (optionally on GPU)
	model = bvcopula.Pair_CopulaGP(bvcopulas,device=device)

	optimizer = torch.optim.Adam([
	    {'params': model.gp_model.mean_module.parameters()},
	    {'params': model.gp_model.variational_strategy.parameters()},
	    {'params': model.gp_model.covar_module.parameters(), 'lr': conf.hyper_lr}, #hyperparameters
	], lr=conf.base_lr)

	# train the model

	mll = VariationalELBO(model.likelihood, model.gp_model,
                            num_data=train_y.size(0))

	losses = torch.zeros(conf.max_num_iter, device=device)
	rbf = torch.zeros(conf.max_num_iter, device=device)
	means = torch.zeros(conf.max_num_iter, device=device)
	nans_detected = 0
	WAIC = -1 #assume that the model will train well
	
	def train(train_x, train_y, num_iter=conf.max_num_iter):
	    model.gp_model.train()
	    model.likelihood.train()

	    p = torch.zeros(1,device=device)
	    nans = torch.zeros(1,device=device)
	    for i in range(num_iter):
	        optimizer.zero_grad()
	        output = model.gp_model(train_x)
	        
	        loss = -mll(output, train_y)  
	 
	        losses[i] = loss.detach()
	        #rbf[i] = model.covar_module.base_kernel.lengthscale.detach()
	        #means[i] = model.variational_strategy.variational_distribution.variational_mean\
	        #		.detach()

	        if len(losses)>100: 
	            p += torch.abs(torch.mean(losses[i-50:i+1]) - torch.mean(losses[i-100:i-50]))

	        if not (i + 1) % conf.iter_print:
	            
	            mean_p = p/100

	            if (0 < mean_p < conf.loss_tol2check_waic):
	                WAIC = model.likelihood.WAIC(model.gp_model(train_x),train_y)
	                if (WAIC > conf.waic_tol):
	                    logging.debug("Training does not look promissing!")
	                    break	

	            if (0 < mean_p < conf.loss_tol):
	                logging.debug("Converged in {} steps!".format(i+1))
	                break
	            p = 0.

	        # The actual optimization step
	        loss.backward(retain_graph=True)
	        optimizer.step()

	t1 = time.time()

	if (len(bvcopulas)!=1) or (bvcopulas[0].name!='Independence'):
		train(train_x,train_y)

	if nans_detected==1:
		logging.warning('NaNs were detected in gradients.')

	if output_loss is not None:
		assert isinstance(output_loss, str)
		plot_loss(output_loss, losses.cpu().numpy(), rbf.cpu().numpy(), means.cpu().numpy())

	if (WAIC < 0): 
	# if model got to the point where it was better than independence: recalculate final WAIC
		WAIC = model.likelihood.WAIC(model.gp_model(train_x),train_y)

	t2 = time.time()
	logging.info('WAIC={:.4f}, took {} sec'.format(WAIC,int(t2-t1)))

	if device!=torch.device('cpu'):
		with torch.cuda.device(device):
			torch.cuda.empty_cache()

	return WAIC, model

def load_model(filename, bvcopulas, device: torch.device, 
	theta_sharing=None):

	logging.info('Loading {}'.format(utils.get_copula_name_string(bvcopulas)))

	# define the model (optionally on GPU)
	model = bvcopula.Pair_CopulaGP(bvcopulas,device=device)

	model.gp_model.load_state_dict(torch.load(filename, map_location=device))
	model.gp_model.eval()

	return model
