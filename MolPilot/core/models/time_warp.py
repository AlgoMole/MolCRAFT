import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat


class Timewarp(nn.Module):
    
    """
    Timewarping proposed in CDCD paper.
    """
    
    def __init__(self, timewarp_type, num_cat_features, num_cont_features,
                 sigma_min_cat, sigma_min_cont, sigma_max_cat, sigma_max_cont, num_bins = 100, decay = 0.5):

        super(Timewarp, self).__init__()
        
        self.timewarp_type = timewarp_type
        self.num_bins = num_bins
        self.num_cat_features = num_cat_features
        self.num_cont_features = num_cont_features
        self.num_features = num_cat_features + num_cont_features

        # combine sigma boundaries for transforming sigmas to [0,1]
        sigma_min = torch.cat((
            torch.tensor(sigma_min_cat).repeat(self.num_cat_features),
            torch.tensor(sigma_min_cont).repeat(self.num_cont_features)
            ), dim = 0)
        sigma_max = torch.cat((
            torch.tensor(sigma_max_cat).repeat(self.num_cat_features),
            torch.tensor(sigma_max_cont).repeat(self.num_cont_features)
            ), dim = 0)
        self.register_buffer('sigma_min', sigma_min)
        self.register_buffer('sigma_max', sigma_max)
        
        
        if timewarp_type == 'single':
            self.num_funcs = 1
        elif timewarp_type == 'bytype':
            self.num_funcs = 2
        elif timewarp_type == 'single_cont':
            self.num_funcs = self.num_cat_features + 1
        elif timewarp_type == 'all':
            self.num_funcs = self.num_cat_features + self.num_cont_features
        
        self.logits_t = nn.Parameter(torch.full((self.num_funcs, num_bins),
                                                -torch.tensor(num_bins).log()))
        self.logits_u = nn.Parameter(torch.full((self.num_funcs, num_bins), 
                                                -torch.tensor(num_bins).log()))

        # copy parameters to keep EMA
        self.decay = decay
        logits_t_shadow = torch.clone(self.logits_t).detach()
        logits_u_shadow = torch.clone(self.logits_u).detach()
        self.register_buffer('logits_t_shadow', logits_t_shadow)
        self.register_buffer('logits_u_shadow', logits_u_shadow)
 

    def update_ema(self):
        with torch.no_grad():
            self.logits_t.copy_(self.decay * self.logits_t_shadow + (1 - self.decay) * self.logits_t.detach())
            self.logits_u.copy_(self.decay * self.logits_u_shadow + (1 - self.decay) * self.logits_u.detach())
            self.logits_t_shadow.copy_(self.logits_t)
            self.logits_u_shadow.copy_(self.logits_u)
     
     
    def get_bins(self, invert, normalize):
         
        if not invert:
            logits_t = self.logits_t
            logits_u = self.logits_u
        else:
            normalize = True
            # we can invert by simply switching the roles of the logits
            logits_t = self.logits_u
            logits_u = self.logits_t
         
        if normalize:
            weights_u = F.softmax(logits_u, dim=1)
        else:
            weights_u = logits_u.exp()
        weights_t = F.softmax(logits_t, dim=1)
        
        # add small constant to each bin size and renormalize
        weights_u = weights_u + 1e-7
        if normalize:
            weights_u = weights_u / weights_u.sum(dim=1, keepdims=True)
        weights_t = weights_t + 1e-7
        weights_t = weights_t / weights_t.sum(dim=1, keepdims=True)

        # get edge values and slopes
        edges_t_right = torch.cumsum(weights_t, dim=1)
        edges_u_right = torch.cumsum(weights_u, dim=1)
        edges_t_left = F.pad(edges_t_right[:, :-1], (1, 0), value=0)
        edges_u_left = F.pad(edges_u_right[:, :-1], (1, 0), value=0)
        slopes = weights_u / weights_t
        
        return edges_t_left, edges_t_right, edges_u_left, edges_u_right, slopes
    
     
    def forward(self, x, invert = False, normalize = False, return_pdf = False):
        
        edges_t_left, edges_t_right, edges_u_left, _, slopes = self.get_bins(invert = invert, normalize = normalize)
       
        if not invert:
            # scale sigmas to [0,1]
            x = (x - self.sigma_min) / (self.sigma_max - self.sigma_min)

            if self.timewarp_type == 'single':
                # all sigmas are the same so just take first one
                input = x[:,0].unsqueeze(0)

            elif self.timewarp_type == 'bytype':
                # first sigma belongs to categorical feature, last to continuous feature
                input = torch.stack((x[:,0], x[:,-1]), dim = 0)
                
            elif self.timewarp_type == 'single_cont':
                # +1 to get one sigma from a continuous feature
                input = x[:, :self.num_cat_features + 1].T
                
            elif self.timewarp_type == 'all':
                input = x.T

        else:
            # in inverse mode: have single u as input, need to repeat u
            input = repeat(x, 'b -> f b', f = self.num_funcs)
 
        bin_idx = torch.searchsorted(edges_t_right, input.contiguous(), right = False)
        bin_idx[bin_idx > self.num_bins - 1] = self.num_bins - 1

        slope = slopes.gather(dim = 1, index = bin_idx) # num_cdfs, batch
        left_t = edges_t_left.gather(dim = 1, index = bin_idx)
        left_u = edges_u_left.gather(dim = 1, index = bin_idx)
        
        if return_pdf:
            return  slope.T.detach()
      
        # linearly interpolate bin edges
        interpolation = (left_u + (input - left_t) * slope).T

        # ensure outputs are in correct dimensions
        if not invert:
            output = interpolation
        else:
            if self.timewarp_type == 'single':
                output = repeat(interpolation, 'b 1 -> b f', f = self.num_features)
            elif self.timewarp_type == 'bytype':
                output = torch.column_stack((
                    repeat(interpolation[:, 0], 'b -> b f', f = self.num_cat_features),
                    repeat(interpolation[:, 1], 'b -> b f', f = self.num_cont_features),
                ))
            elif self.timewarp_type == 'single_cont':
                output = torch.column_stack((
                interpolation[:, :self.num_cat_features],
                repeat(interpolation[:, -1], 'b -> b f', f = self.num_cont_features)
                ))
            elif self.timewarp_type == 'all':
                output = interpolation

        if not return_pdf and normalize:
            output = torch.clamp(output, 0, 1)

        if invert:
            output = output * (self.sigma_max - self.sigma_min) + self.sigma_min

        return output
        
    
    def loss_fn(self, sigmas, losses):
        
        if self.timewarp_type == 'single':
            # fit average loss (over all feature)
            losses = losses.mean(1, keepdim=True) # (B,1)
        elif self.timewarp_type == 'bytype':
            # fit average loss over cat and over cont features separately
            losses_cat = losses[:, :self.num_cat_features].mean(1, keepdim=True) # (B,1)
            losses_cont = losses[:, self.num_cat_features:].mean(1, keepdim=True) # (B,1)
            losses = torch.cat((losses_cat, losses_cont), dim=1)
        elif self.timewarp_type == 'single_cont':
            # fit average loss over cont features and feature-specific losses for cat features
            losses_cat = losses[:, :self.num_cat_features] # (B, num_cat_features)
            losses_cont = losses[:, self.num_cat_features:].mean(1, keepdim=True) # (B,1)
            losses = torch.cat((losses_cat, losses_cont), dim=1)

        # F(t)
        losses_estimated = self.forward(sigmas)
        
        
        with torch.no_grad():
            pdf = self.forward(sigmas, return_pdf = True, normalize = True).detach()

        return ((losses_estimated - losses) ** 2) / pdf
    


class Timewarp_Logistic(nn.Module):
    """
    Our version for timewarping with exact cdfs instead of p.w.l. functions.
    As cdf we use a domain-adapted cdf of the logistic distribution.
    """
    
    def __init__(self, timewarp_type, num_cat_features, num_cont_features, sigma_min, sigma_max, weight_low_noise=1.0, decay=0.5):

        super(Timewarp_Logistic, self).__init__()
        
        self.timewarp_type = timewarp_type
        self.num_cat_features = num_cat_features
        self.num_cont_features = num_cont_features
        self.num_features = num_cat_features + num_cont_features
        
        # save bounds for min max scaling
        self.register_buffer('sigma_min', sigma_min)
        self.register_buffer('sigma_max', sigma_max)
        
        if timewarp_type == 'single':
            self.num_funcs = 1
        elif timewarp_type == 'bytype':
            self.num_funcs = 2
        elif timewarp_type == 'single_cont':
            self.num_funcs = self.num_cat_features + 1
        elif timewarp_type == 'all':
            self.num_funcs = self.num_cat_features + self.num_cont_features

        # init parameters
        v = torch.tensor(1.01)
        logit_v = torch.log(torch.exp(v-1) - 1)
        self.logits_v = nn.Parameter(torch.full((self.num_funcs,), fill_value=logit_v))
        self.register_buffer("init_v", self.logits_v.clone())

        p_large_noise = torch.tensor(1 / (weight_low_noise + 1))
        logit_mu = torch.log(((1 / (1-p_large_noise)) - 1)) / v
        self.logits_mu = nn.Parameter(torch.full((self.num_funcs,), fill_value=logit_mu))
        self.register_buffer("init_mu", self.logits_mu.clone())
        
        # init gamma, scaling parameter to 1
        self.logits_gamma = nn.Parameter((torch.ones((self.num_funcs, 1)).exp()-1).log()) 

        # for ema
        self.decay = decay
        logits_v_shadow = torch.clone(self.logits_v).detach()
        logits_mu_shadow = torch.clone(self.logits_mu).detach()
        logits_gamma_shadow = torch.clone(self.logits_gamma).detach()
        self.register_buffer('logits_v_shadow', logits_v_shadow)
        self.register_buffer('logits_mu_shadow', logits_mu_shadow)
        self.register_buffer('logits_gamma_shadow', logits_gamma_shadow)


    def update_ema(self):
        with torch.no_grad():
            self.logits_v.copy_(self.decay * self.logits_v_shadow + (1 - self.decay) * self.logits_v.detach())
            self.logits_mu.copy_(self.decay * self.logits_mu_shadow + (1 - self.decay) * self.logits_mu.detach())
            self.logits_gamma.copy_(self.decay * self.logits_gamma_shadow + (1 - self.decay) * self.logits_gamma.detach())
            self.logits_v_shadow.copy_(self.logits_v)
            self.logits_mu_shadow.copy_(self.logits_mu)
            self.logits_gamma_shadow.copy_(self.logits_gamma)
     
     

    def get_params(self):
        logit_mu = self.logits_mu # let underlying parameter be ln(mu / (1-mu))
        v = 1 + F.softplus(self.logits_v) # v > 1
        scale = F.softplus(self.logits_gamma)
        return logit_mu, v, scale
    

    def cdf_fn(self, x, logit_mu, v):
        "mu in (0,1), v >= 1"
        Z = ((x / (1-x)) / logit_mu.exp()) ** (-v)
        return 1 / (1 + Z)
    

    def pdf_fn(self, x, logit_mu, v):
        Z = ((x / (1-x)) / logit_mu.exp()) ** (-v)
        return (v / (x * (1-x))) * (Z / ((1 + Z) ** 2))

    
    def quantile_fn(self, u, logit_mu, v):
        c = logit_mu + 1/v * torch.special.logit(u, eps=1e-7)
        return F.sigmoid(c)
    
     
    def forward(self, x, invert = False, normalize = False, return_pdf = False):
        
        logit_mu, v, scale = self.get_params()
  
        if not invert:
            
            if normalize:
                scale = 1.0

            # can have more sigmas than cdfs
            x = (x - self.sigma_min) / (self.sigma_max - self.sigma_min)
            
            # ensure x is never 0 or 1 to ensure robustness
            x = torch.clamp(x, 1e-7, 1-1e-7)
        
            if self.timewarp_type == 'single':
                # all sigmas are the same so just take first one
                input = x[:,0].unsqueeze(0)
        
            elif self.timewarp_type == 'bytype':
                # first sigma belongs to categorical feature, last to continuous feature
                input = torch.stack((x[:,0], x[:,-1]), dim=0)
                
            elif self.timewarp_type == 'single_cont':
                # +1 to get one sigma from a continuous feature
                input = x[:, :self.num_cat_features + 1].T
                
            elif self.timewarp_type == 'all':
                input = x.T # (num_features, batch)
                
            if return_pdf:
                output = (torch.vmap(self.pdf_fn, in_dims=0)(input, logit_mu, v)).T
            else:
                output = (torch.vmap(self.cdf_fn, in_dims=0)(input, logit_mu, v) * scale).T
            
        else:
            # have single u, need to repeat u
            input = repeat(x, 'b -> f b', f = self.num_funcs)
            output = (torch.vmap(self.quantile_fn, in_dims=0)(input, logit_mu, v)).T
            
            if self.timewarp_type == 'single':
                output = repeat(output, 'b 1 -> b f', f = self.num_features)
            elif self.timewarp_type == 'bytype':
                output = torch.column_stack((
                    repeat(output[:, 0], 'b -> b f', f = self.num_cat_features),
                    repeat(output[:, 1], 'b -> b f', f = self.num_cont_features),
                ))
            elif self.timewarp_type == 'single_cont':
                output = torch.column_stack((
                output[:, :self.num_cat_features],
                repeat(output[:, -1], 'b -> b f', f = self.num_cont_features)
                ))
                
            zero_mask = (x == 0.0)
            one_mask = (x == 1.0)
            output[zero_mask] = 0.0
            output[one_mask] = 1.0
            
            output = output * (self.sigma_max - self.sigma_min) + self.sigma_min
            
        return output


    def loss_fn(self, sigmas, losses):
        
        # losses and sigmas have shape (B, num_features)
        
        if self.timewarp_type == 'single':
            # fit average loss (over all feature)
            losses = losses.mean(1, keepdim=True) # (B,1)
        elif self.timewarp_type == 'bytype':
            # fit average loss over cat and over cont features separately
            losses_cat = losses[:, :self.num_cat_features].mean(1, keepdim=True) # (B,1)
            losses_cont = losses[:, self.num_cat_features:].mean(1, keepdim=True) # (B,1)
            losses = torch.cat((losses_cat, losses_cont), dim=1)
        elif self.timewarp_type == 'single_cont':
            # fit average loss over cont features and feature-specific losses for cat features
            losses_cat = losses[:, :self.num_cat_features] # (B, num_cat_features)
            losses_cont = losses[:, self.num_cat_features:].mean(1, keepdim=True) # (B,1)
            losses = torch.cat((losses_cat, losses_cont), dim=1)
            
        losses_estimated = self.forward(sigmas)
        
        with torch.no_grad():
            pdf = self.forward(sigmas, return_pdf = True).detach()
        
        return ((losses_estimated - losses) ** 2) / (pdf + 1e-7)


