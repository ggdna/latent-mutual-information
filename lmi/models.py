import numpy as np
import torch 
import torch.nn as nn

class AECross(nn.Module):
    
    def __init__(self, x_dim, y_dim, latent_size, alpha=1, lam=1):
        """

        Paired AE models with cross predictive regularization
        
        """
        
        super(AECross, self).__init__()
        
        # choosing hidden layer sizes
        # Lx = int(2**np.floor(np.log2(x_dim)))
        # Ly = int(2**np.floor(np.log2(y_dim)))
        
        Lx, Ly = 1024, 1024
        
        if x_dim > 2048:
            Lx = int(2**np.floor(np.log2(x_dim)))
        if y_dim > 2048:
            Ly = int(2**np.floor(np.log2(y_dim)))

        self.x_encoder = nn.Sequential(nn.Linear(x_dim, Lx),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx, Lx//2),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx//2, Lx//4),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx//4, latent_size),
                                      nn.LeakyReLU(negative_slope=0.2))
        
        self.y_encoder = nn.Sequential(nn.Linear(y_dim, Ly),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly, Ly//2),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly//2, Ly//4),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly//4, latent_size),
                                      nn.LeakyReLU(negative_slope=0.2))
        
        self.xx_decoder = nn.Sequential(nn.Linear(latent_size, Lx//4),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx//4, Lx//2),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx//2, Lx),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx, x_dim),)
                                    #   nn.LeakyReLU(negative_slope=0.2))
        
        self.yy_decoder = nn.Sequential(nn.Linear(latent_size, Ly//4),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly//4, Ly//2),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly//2, Ly),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly, y_dim),)
                                    #   nn.LeakyReLU(negative_slope=0.2))
        
        self.yx_decoder = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Linear(latent_size, Lx//4),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(Lx//4, Lx//2),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(Lx//2, Lx),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(Lx, x_dim),)
                                    #   nn.LeakyReLU(negative_slope=0.2))
        
        self.xy_decoder = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Linear(latent_size, Ly//4),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(Ly//4, Ly//2),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(Ly//2, Ly),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(Ly, y_dim),)
                                    #   nn.LeakyReLU(negative_slope=0.2))

        self.alpha = alpha
        self.lam = lam

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        for net in [self.x_encoder, self.y_encoder, self.xx_decoder, self.yy_decoder, 
                    self.xy_decoder, self.yx_decoder]:
            net.apply(init_weights)
        
    def encode(self, x_samples, y_samples):
        Zx = self.x_encoder(x_samples)
        Zy = self.y_encoder(y_samples)
        return Zx, Zy

    def cross_decode(self, Zx, Zy):
        y_hat = self.xy_decoder(Zx)
        x_hat = self.yx_decoder(Zy)
        return x_hat, y_hat
    
    def decode(self, Zx, Zy):
        x_hat = self.xx_decoder(Zx)
        y_hat = self.yy_decoder(Zy)
        return x_hat, y_hat
    
    def rec_loss(self, hat, samples):
        return torch.nn.functional.mse_loss(hat, samples, reduction='mean')

    def learning_loss(self, x_samples, y_samples):
        
        Zx, Zy = self.encode(x_samples, y_samples)
        Xh, Yh = self.decode(Zx, Zy)
        cXh, cYh = self.cross_decode(Zx, Zy)
        
        auto_loss = self.rec_loss(Xh, x_samples) + self.rec_loss(Yh, y_samples)
        cross_loss = self.rec_loss(cXh, x_samples) + self.rec_loss(cYh, y_samples)
        # print(auto_loss, cross_loss)
        
        return self.lam*cross_loss + self.alpha*auto_loss


class AEMINE(nn.Module):
    
    def __init__(self, x_dim, y_dim, latent_size, alpha=1, lam=1):
        """

        Paired AE models with MINE regularization

        """
        
        super(AEMINE, self).__init__()
        
        # choosing hidden layer sizes
        # Lx = int(2**np.floor(np.log2(x_dim)))
        # Ly = int(2**np.floor(np.log2(y_dim)))
        Lx, Ly = 1024, 1024
        
        self.x_encoder = nn.Sequential(nn.Linear(x_dim, Lx),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx, Lx//2),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx//2, Lx//4),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx//4, latent_size),
                                      nn.LeakyReLU(negative_slope=0.2))
        
        self.y_encoder = nn.Sequential(nn.Linear(y_dim, Ly),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly, Ly//2),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly//2, Ly//4),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly//4, latent_size),
                                      nn.LeakyReLU(negative_slope=0.2))
        
        self.xx_decoder = nn.Sequential(nn.Linear(latent_size, Lx//4),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx//4, Lx//2),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx//2, Lx),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx, x_dim),)
                                    #   nn.LeakyReLU(negative_slope=0.2))
        
        self.yy_decoder = nn.Sequential(nn.Linear(latent_size, Ly//4),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly//4, Ly//2),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly//2, Ly),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly, y_dim),)
                                    #   nn.LeakyReLU(negative_slope=0.2))
        
        self.T_func = nn.Sequential(nn.Linear(latent_size*2, latent_size),
                                    nn.ReLU(),
                                    nn.Linear(latent_size, 1))

        self.alpha = alpha
        self.lam = lam

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        for net in [self.x_encoder, self.y_encoder, self.xx_decoder, self.yy_decoder, 
                    self.T_func]:
            net.apply(init_weights)
        
    def encode(self, x_samples, y_samples):
        Zx = self.x_encoder(x_samples)
        Zy = self.y_encoder(y_samples)
        return Zx, Zy

    def MINELoss(self, x_samples, y_samples):  
        
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())
        return -lower_bound
    
    def decode(self, Zx, Zy):
        x_hat = self.xx_decoder(Zx)
        y_hat = self.yy_decoder(Zy)
        return x_hat, y_hat
    
    def rec_loss(self, hat, samples):
        return torch.nn.functional.mse_loss(hat, samples, reduction='mean')

    def learning_loss(self, x_samples, y_samples):
        
        Zx, Zy = self.encode(x_samples, y_samples)
        Xh, Yh = self.decode(Zx, Zy)
        
        auto_loss = self.rec_loss(Xh, x_samples) + self.rec_loss(Yh, y_samples)
        cross_loss = self.MINELoss(Zx, Zy)
        
        return self.lam*cross_loss + self.alpha*auto_loss


class AEInfoNCE(nn.Module):
    
    def __init__(self, x_dim, y_dim, latent_size, alpha=1, lam=1):
        """

        Paired AE models with InfoNCE regularization
        
        """
        
        super(AEInfoNCE, self).__init__()
        
        # choosing hidden layer sizes
        # Lx = int(2**np.floor(np.log2(x_dim)))
        # Ly = int(2**np.floor(np.log2(y_dim)))
        Lx, Ly = 1024, 1024
        
        self.x_encoder = nn.Sequential(nn.Linear(x_dim, Lx),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx, Lx//2),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx//2, Lx//4),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx//4, latent_size),
                                      nn.LeakyReLU(negative_slope=0.2))
        
        self.y_encoder = nn.Sequential(nn.Linear(y_dim, Ly),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly, Ly//2),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly//2, Ly//4),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly//4, latent_size),
                                      nn.LeakyReLU(negative_slope=0.2))
        
        self.xx_decoder = nn.Sequential(nn.Linear(latent_size, Lx//4),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx//4, Lx//2),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx//2, Lx),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Lx, x_dim),)
                                    #   nn.LeakyReLU(negative_slope=0.2))
        
        self.yy_decoder = nn.Sequential(nn.Linear(latent_size, Ly//4),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly//4, Ly//2),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly//2, Ly),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(Ly, y_dim),)
                                    #   nn.LeakyReLU(negative_slope=0.2))
        
        self.F_func = nn.Sequential(nn.Linear(latent_size+latent_size, latent_size),
                                    nn.ReLU(),
                                    nn.Linear(latent_size, 1),
                                    nn.Softplus())

        self.alpha = alpha
        self.lam = lam

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        for net in [self.x_encoder, self.y_encoder, self.xx_decoder, self.yy_decoder, 
                    self.F_func]:
            net.apply(init_weights)
        
    def encode(self, x_samples, y_samples):
        Zx = self.x_encoder(x_samples)
        Zy = self.y_encoder(y_samples)
        return Zx, Zy

    def InfoNCELoss(self, x_samples, y_samples):  
        
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        return -lower_bound
    
    def decode(self, Zx, Zy):
        x_hat = self.xx_decoder(Zx)
        y_hat = self.yy_decoder(Zy)
        return x_hat, y_hat
    
    def rec_loss(self, hat, samples):
        return torch.nn.functional.mse_loss(hat, samples, reduction='mean')

    def learning_loss(self, x_samples, y_samples):
        
        Zx, Zy = self.encode(x_samples, y_samples)
        Xh, Yh = self.decode(Zx, Zy)
        
        auto_loss = self.rec_loss(Xh, x_samples) + self.rec_loss(Yh, y_samples)
        cross_loss = self.InfoNCELoss(Zx, Zy)
        
        return self.lam*cross_loss + self.alpha*auto_loss
