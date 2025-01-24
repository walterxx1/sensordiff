import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

parent_dir = os.path.join(current_dir, '../..')
sys.path.append(parent_dir)

import torch
import pdb
# from Models.models import *
from Models.models_new import *
from functools import partial
from inspect import isfunction
from tqdm import tqdm

from TimeMixer.models.TimeMixer import Model as TimeMixer

from .interpretable_diffusion.transformer import Transformer

"""
version1: 
simple DDPM, unconditional model, when sampling, directly generate
sensor datas from normal distribution
problem: can't decide the specific sample, need to change this into conditional diff

version2: 
conditional model
model the task into forecasting question
"""

class VarianceSchedule(nn.Module):
    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            # betas = self.linear_beta_schedule(timesteps=num_steps)
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            betas = self.cosine_beta_schedule(timesteps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        # log_alphas = torch.log(alphas)
        # for i in range(1, log_alphas.size(0)):  # 1 to T
        #     log_alphas[i] += log_alphas[i - 1]
        # alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def linear_beta_schedule(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)

    def cosine_beta_schedule(self, timesteps, s = 0.008):
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

class DiffSensor(nn.Module):
    def __init__(self, config, args, device, criterion, iftest=False) -> None:
        super().__init__()
        self.config = config
        self.args = args
        self.criterion = criterion
        
        config_model = config['model']
        ctx_dim = config_model['context_dim']
        p_dim = config_model['point_dim']
        ctx_dim_out = config_model['context_dim_out']
        feat_dim = config_model['feature_dim']
        seq_length = config_model['seq_length']
        prior_len = config['dataloader']['prior_size']

        self.model = TimeDiffBackbone(p_dim, feat_dim, seq_length, iftest).to(device)
        # self.model = UnetConcatLinear(ctx_dim, p_dim, ctx_dim_out, feat_dim, prior_len).to(device)
        self.device = device
        self.var_sched = VarianceSchedule(config['model']['diff_num_steps'])

        self.model_unet = VanillaUnet(ctx_dim).to(device)
        
    def _get_loss_nodiff(self, context, future, t=None):
        output = self.model_unet(context)
        loss = self.criterion(output, future)
        return loss
    
    def _get_sample_nodiff(self, context):
        return self.model_unet(context).cpu()
    
    def norm_rand(self, tensor, min_value=-1.0, max_value=1.0):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
        
        scaled_tensor = scaled_tensor * (max_value - min_value) + min_value
        
        return scaled_tensor
    
    def denorm_rand(self, tensor, ori_min, ori_max, curr_range=(-1,1)):
        curr_min, curr_max = curr_range
        denormalized_cur = (tensor - curr_min) / (curr_max - curr_min)
        return denormalized_cur * (ori_max - ori_min) + ori_min
    
    def get_loss(self, context, future, t=None):
        
        batch_size, _, _ = future.size()
        if t is None:
            t = self.var_sched.uniform_sample_t(batch_size)
        
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].to(self.device)
        
        c0 = torch.sqrt(alpha_bar).view(-1,1,1).to(self.device)
        c1 = torch.sqrt(1 - alpha_bar).view(-1,1,1).to(self.device)
        
        e_rand = torch.randn_like(future).to(self.device)
        # scale the random tensor into (0,1)
        # e_rand = self.norm_rand(e_rand)
        # pdb.set_trace()
        

        # print('check c0 c1 shape', c0.shape, c1.shape)
        # e_theta = self.model(c0 * x + c1 * e_rand, beta=beta)
        noise_future = c0 * future + c1 * e_rand
        e_theta = self.model(beta, context, noise_future)
        loss = self.criterion(e_theta, e_rand)
        # loss = F.mse_loss(e_theta, e_rand, reduction='mean')
        
        return loss
    
    def sample(self, context, sampling='ddpm'):
        num_points = self.config['dataloader']['future_size']
        point_dim = self.config['model']['point_dim']
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(self.device)
        # x_T = self.norm_rand(x_T)
        diffsensor = {self.var_sched.num_steps: x_T}
        stride = self.config['model']['stride']
        total_mse_loss = 0.0
        # print('check numsteps', self.var_sched.num_steps, stride)
        steps = 0
        for t in range(self.var_sched.num_steps, 0, -stride):
            # z is a random noise
            if t > 1:
                z = torch.randn_like(x_T)
                # z = self.norm_rand(z)
            else:
                z = torch.zeros_like(x_T)

            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_next = self.var_sched.alpha_bars[t-stride]
            sigma = self.var_sched.get_sigmas(t, 0.0)
            
            # sigma = torch.clamp(sigma, min=0.01, max=1.0)
            
            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            
            x_t = diffsensor[t]
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.model(beta, context, x_t)
            # pdb.set_trace()
            
            if sampling == 'ddpm':
                x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            elif sampling == 'ddim':
                x0_t = (x_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * e_theta
            else:
                # python debug
                pdb.set_trace()
                
            reconstuct_noise = (x_t - c0 * x_next) / (c1 + 1e-8)
            noise_loss = F.mse_loss(e_theta, reconstuct_noise, reduction='mean')
            total_mse_loss += noise_loss.item()
                
            diffsensor[t-stride] = x_next.detach()
            diffsensor[t] = diffsensor[t].cpu()
            steps += 1
            # del diffsensor[t]
        
        return diffsensor[0], total_mse_loss / steps



"""
From gpt o1
"""
class VarianceSchedule_gpt:
    def __init__(self, num_steps, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_steps = num_steps
        self.device = device
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Create a linear schedule for betas
        self.betas = torch.linspace(beta_start, beta_end, steps=num_steps+1).to(self.device)  # t from 0 to num_steps
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def sample_timesteps(self, batch_size):
        return torch.randint(1, self.num_steps+1, (batch_size,), device=self.device)  # t in [1, num_steps]

# Diffusion Process Class with Sampling Method
class DiffusionProcess(nn.Module):
    def __init__(self, config, args, device, criterion, iftest=False):
        super(DiffusionProcess, self).__init__()
        
        config_model = config['model']
        ctx_dim = config_model['context_dim']
        p_dim = config_model['point_dim']
        ctx_dim_out = config_model['context_dim_out']
        feat_dim = config_model['feature_dim']
        seq_length = config_model['seq_length']
        num_steps = config_model['diff_num_steps']
        
        prior_len = config['dataloader']['prior_size']
        
        self.backbone = TimeSeriesBackbone(p_dim, feat_dim, ctx_dim_out, time_emb_dim=128).to(device)
        # self.backbone = UnetConcatLinear(ctx_dim, p_dim, ctx_dim_out, feat_dim, seq_length).to(device)
        self.variance_schedule = VarianceSchedule_gpt(num_steps, device=device)
        self.device = device
        
    def get_loss(self, context, future):
        batch_size = context.size(0)
        t = self.variance_schedule.sample_timesteps(batch_size)  # t in [1, num_steps]
        
        # Get corresponding alpha values
        alpha_bars_t = self.variance_schedule.alpha_bars[t].to(self.device)         # Shape: (batch_size,)
        sqrt_alpha_bars_t = torch.sqrt(alpha_bars_t).view(-1, 1, 1)                 # Shape: (batch_size, 1, 1)
        sqrt_one_minus_alpha_bars_t = torch.sqrt(1 - alpha_bars_t).view(-1, 1, 1)   # Shape: (batch_size, 1, 1)
        
        # Sample random noise epsilon
        epsilon = torch.randn_like(future).to(self.device)
        
        # Generate x_t from future data and noise
        x_t = sqrt_alpha_bars_t * future + sqrt_one_minus_alpha_bars_t * epsilon
        
        # Predict noise using the backbone model
        epsilon_theta = self.backbone(context, x_t, t)
        # betas = self.variance_schedule.betas[t].to(self.device)
        # epsilon_theta = self.backbone(betas, context, x_t)
        
        # print('======= check shape', epsilon_theta.shape, epsilon.shape)
        # Compute MSE loss between the true noise and the predicted noise
        loss = F.mse_loss(epsilon_theta, epsilon)
        return loss
    
    def sample(self, context, num_steps=None):
        if num_steps is None:
            num_steps = self.variance_schedule.num_steps
        
        batch_size, seq_length, input_channels = context.size()
        device = self.device
        
        # Start from pure noise
        x_t = torch.randn(batch_size, seq_length, input_channels).to(device)
        
        total_noise_loss = 0.0
        num_iterations = 0
        
        # Iteratively sample from t = num_steps to t = 1
        for t in reversed(range(1, num_steps + 1)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get beta_t, alpha_t, and alpha_bar_t
            beta_t = self.variance_schedule.betas[t].to(device)
            alpha_t = self.variance_schedule.alphas[t].to(device)
            alpha_bar_t = self.variance_schedule.alpha_bars[t].to(device)
            alpha_bar_t_minus1 = self.variance_schedule.alpha_bars[t - 1].to(device) if t > 1 else torch.tensor(1.0).to(device)
            
            # Predict noise using the backbone model
            epsilon_theta = self.backbone(context, x_t, t_batch)
            # betas = self.variance_schedule.betas[[t]*batch_size].to(self.device)
            # epsilon_theta = self.backbone(betas, context, x_t)
            
            # Compute the mean (mu) of q(x_{t-1} | x_t, x_0)
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)
            mu = coef1 * (x_t - coef2 * epsilon_theta)
            
            if t > 1:
                # Sample from normal distribution with mean mu and variance sigma_t^2
                sigma_t = torch.sqrt(beta_t)
                noise = torch.randn_like(x_t).to(device)
                x_prev = mu + sigma_t * noise
                
                epsilon_actual = noise
                
                noise_loss = F.mse_loss(epsilon_theta, epsilon_actual, reduction='mean')
                total_noise_loss += noise_loss.item()
                num_iterations += 1
            else:
                # For t == 1, return mu as the final sample
                x_prev = mu
            x_t = x_prev
            
        if num_iterations > 0:
            avg_noise_loss = total_noise_loss / num_iterations
        else:
            avg_noise_loss = None
        # The final sample x_0
        return x_t, avg_noise_loss



class GaussianProcess_transformer(nn.Module):
    def __init__(self, config, device):
        super(GaussianProcess_transformer, self).__init__()
        
        config_model = config['model']
        ctx_dim = config_model['context_dim']
        p_dim = config_model['point_dim']
        ctx_dim_out = config_model['context_dim_out']
        feat_dim = config_model['feature_dim']
        seq_length = config_model['seq_length']
        self.num_steps = config_model['diff_num_steps']
        self.stride = config_model['stride']

        # prior_len = config['dataloader']['prior_size']
        
        # self.model = UNetGenerate(
        #     in_channels=p_dim,
        #     out_channels=p_dim,
        # ).to(device)
        
        feature_size = 6
        n_layer_enc = 3
        n_layer_dec = 2
        n_heads = 4
        attn_pd = 0.0
        resid_pd = 0.0
        mlp_hidden_times = 4
        d_model = 64
        kernel_size = 1
        padding_size = 0
        
        
        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size]).to(device)
        
        
        
        self.num_steps = config_model['diff_num_steps']
        self.stride = config_model['stride']        
        self.input_dim = p_dim
        self.seq_length = seq_length
        self.device = device
        
        # beta_start, beta_end = 1e-4, 2e-2
        scale = 1000 / self.num_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        # Create a linear schedule for betas
        self.betas = torch.linspace(beta_start, beta_end, steps=self.num_steps+1).to(self.device)  # t from 0 to num_steps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a: torch.FloatTensor, t: torch.LongTensor, x_shape):
        # get the param of given timestep t
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    
    def q_sample(self, x_start: torch.FloatTensor, t: torch.LongTensor, noise=None):
        # forward diffusion (using the nice property): q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_mean_variance(self, x_start: torch.FloatTensor, t: torch.LongTensor):
        # Get the mean and variance of q(x_t | x_0).
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start: torch.FloatTensor, x_t: torch.FloatTensor, t: torch.LongTensor):
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        posterior_mean = self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_start_from_noise(self, x_t: torch.FloatTensor, t: torch.LongTensor, noise: torch.FloatTensor):
        # compute x_0 from x_t and pred noise: the reverse of `q_sample`
        return self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    """
    predict noise from start
    start is predicted by the model
    """
    def predict_noise_from_start(self, x_t: torch.FloatTensor, t: torch.LongTensor, x_start: torch.FloatTensor):
        return (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_mean_variance(self, context, x_t: torch.FloatTensor, t: torch.LongTensor, clip_denoised=True):
        # compute predicted mean and variance of p(x_{t-1} | x_t)
        # predict noise using model
        pred_noise = self.model(context, x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1.0, max=1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, context, x_t: torch.FloatTensor, t: torch.LongTensor, clip_denoised=True):
        # denoise_step: sample x_{t-1} from x_t and pred_noise
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(context, x_t, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t).to(self.device)
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    @torch.no_grad()
    def sample_generate(self, label, num_steps=None):
        batch_size = label.shape[0]
        img = torch.randn((batch_size, self.seq_length, self.input_dim), device=self.device)
        context = label
        img = img.permute(0, 2, 1)
        imgs = []
        for i in reversed(range(0, self.num_steps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(context, img, t)
            imgs.append(img.cpu().numpy())
        img = img.permute(0, 2, 1)
        return imgs, img
    
    def get_loss_generate(self, label, future):
        future = future.permute(0, 2, 1)
        batch_size = label.shape[0]
        t = torch.randint(0, self.num_steps+1, (batch_size,), device=self.device)
        noise = torch.randn_like(future)  # random noise ~ N(0, 1)
        x_noisy = self.q_sample(future, t, noise=noise)  # x_t ~ q(x_t | x_0)
        # pdb.set_trace()
        predicted_noise = self.model(label, x_noisy, t)  # predict noise from noisy image
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    @torch.no_grad()
    # def sample(self,  image_size, batch_size=8, channels=3):
    def sample(self, context, num_steps=None):
        # denoise: reverse diffusion
        # shape = (batch_size, channels, image_size, image_size)
        # start from pure noise (for each example in the batch)
        # img = torch.randn(shape, device=self.device)  # x_T ~ N(0, 1)
        batch_size, seq_length, input_channels = context.size()
        img = torch.randn((batch_size, seq_length, input_channels), device=self.device)
        context, img = context.permute(0, 2, 1), img.permute(0, 2, 1)
        imgs = []
        # for i in tqdm(reversed(range(0, self.num_steps)), desc="sampling loop time step", total=self.num_steps):
        for i in reversed(range(0, self.num_steps, self.stride)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(context, img, t)
            # imgs.append(img.cpu().numpy())
        img = img.permute(0, 2, 1)
        return imgs, img
    
    
    # def train_losses(self, model, x_start: torch.FloatTensor, t: torch.LongTensor):
    def get_loss(self, context, future):
        context, future = context.permute(0, 2, 1), future.permute(0, 2, 1)
        # compute train losses
        batch_size = context.size(0)
        t = torch.randint(0, self.num_steps+1, (batch_size,), device=self.device)
        noise = torch.randn_like(future)  # random noise ~ N(0, 1)
        x_noisy = self.q_sample(future, t, noise=noise)  # x_t ~ q(x_t | x_0)
        # print('diff_477', context.shape, x_noisy.shape, t.shape)
        predicted_noise = self.model(context, x_noisy, t)  # predict noise from noisy image
        loss = F.mse_loss(noise, predicted_noise)
        return loss


    def moveing_avg_filter(self, tensor, window_size):
        # padding_size = (window_size - 1) // 2
        padding_left = (window_size - 1) // 2
        padding_right = window_size // 2
        
        # padded_tensor = torch.nn.functional.pad(tensor, (padding_size, padding_size), mode='reflect')
        padded_tensor = torch.nn.functional.pad(tensor, (padding_left, padding_right), mode='reflect')
        
        unfolded = padded_tensor.unfold(dimension=2, size=window_size, step=1)
        smoothed_tensor = unfolded.mean(dim=-1)
        return smoothed_tensor


# Diffusion Process Class with Sampling Method
class GaussianProcess(nn.Module):
    def __init__(self, config, device):
        super(GaussianProcess, self).__init__()
        
        config_model = config['model']
        ctx_dim = config_model['context_dim']
        p_dim = config_model['point_dim']
        ctx_dim_out = config_model['context_dim_out']
        feat_dim = config_model['feature_dim']
        seq_length = config_model['seq_length']
        self.num_steps = config_model['diff_num_steps']
        self.stride = config_model['stride']

        # prior_len = config['dataloader']['prior_size']
        
        self.model = UNetGenerate(
            in_channels=p_dim,
            out_channels=p_dim,
        ).to(device)
        
        self.fc_mu = nn.Linear(p_dim*seq_length, 128).to(device)
        self.fc_var = nn.Linear(p_dim*seq_length, 128).to(device)
        
        self.input_dim = p_dim
        self.seq_length = seq_length
        self.device = device
        
        # beta_start, beta_end = 1e-4, 2e-2
        scale = 1000 / self.num_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        # Create a linear schedule for betas
        self.betas = torch.linspace(beta_start, beta_end, steps=self.num_steps+1).to(self.device)  # t from 0 to num_steps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a: torch.FloatTensor, t: torch.LongTensor, x_shape):
        # get the param of given timestep t
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    
    def q_sample(self, x_start: torch.FloatTensor, t: torch.LongTensor, noise=None):
        # forward diffusion (using the nice property): q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_mean_variance(self, x_start: torch.FloatTensor, t: torch.LongTensor):
        # Get the mean and variance of q(x_t | x_0).
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start: torch.FloatTensor, x_t: torch.FloatTensor, t: torch.LongTensor):
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        posterior_mean = self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_start_from_noise(self, x_t: torch.FloatTensor, t: torch.LongTensor, noise: torch.FloatTensor):
        # compute x_0 from x_t and pred noise: the reverse of `q_sample`
        return self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    """
    predict noise from start
    start is predicted by the model
    """
    def predict_noise_from_start(self, x_t: torch.FloatTensor, t: torch.LongTensor, x_start: torch.FloatTensor):
        return (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_mean_variance(self, context, x_t: torch.FloatTensor, t: torch.LongTensor, clip_denoised=True):
        # compute predicted mean and variance of p(x_{t-1} | x_t)
        # predict noise using model
        pred_noise = self.model(context, x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1.0, max=1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, context, x_t: torch.FloatTensor, t: torch.LongTensor, clip_denoised=True):
        # denoise_step: sample x_{t-1} from x_t and pred_noise
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(context, x_t, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t).to(self.device)
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    @torch.no_grad()
    def sample_generate(self, label, num_steps=None):
        batch_size = label.shape[0]
        img = torch.randn((batch_size, self.seq_length, self.input_dim), device=self.device)
        context = label
        img = img.permute(0, 2, 1)
        imgs = []
        for i in reversed(range(0, self.num_steps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(context, img, t)
            imgs.append(img.cpu().numpy())
        img = img.permute(0, 2, 1)
        return imgs, img
    
    def get_loss_generate(self, label, future):
        future = future.permute(0, 2, 1)
        batch_size = label.shape[0]
        t = torch.randint(0, self.num_steps+1, (batch_size,), device=self.device)
        noise = torch.randn_like(future)  # random noise ~ N(0, 1) # (b, c, t)
        x_noisy = self.q_sample(future, t, noise=noise)  # x_t ~ q(x_t | x_0)
        # pdb.set_trace()
        predicted_noise = self.model(label, x_noisy, t)  # predict noise from noisy image # (b, c, t)
        
        mu_pred = self.fc_mu(predicted_noise.permute(0, 2, 1).reshape(batch_size, -1))
        logvar_pred = self.fc_var(predicted_noise.permute(0, 2, 1).reshape(batch_size, -1))
        kl_loss = -0.5 * torch.sum(1 + logvar_pred - mu_pred.pow(2) - logvar_pred.exp())
        
        mse_loss = F.mse_loss(noise, predicted_noise)
        
        loss = mse_loss + 0.01 * kl_loss
        # pdb.set_trace()
        return loss
    
    @torch.no_grad()
    # def sample(self,  image_size, batch_size=8, channels=3):
    def sample(self, context, num_steps=None):
        # denoise: reverse diffusion
        # shape = (batch_size, channels, image_size, image_size)
        # start from pure noise (for each example in the batch)
        # img = torch.randn(shape, device=self.device)  # x_T ~ N(0, 1)
        batch_size, seq_length, input_channels = context.size()
        img = torch.randn((batch_size, seq_length, input_channels), device=self.device)
        context, img = context.permute(0, 2, 1), img.permute(0, 2, 1)
        imgs = []
        # for i in tqdm(reversed(range(0, self.num_steps)), desc="sampling loop time step", total=self.num_steps):
        for i in reversed(range(0, self.num_steps, self.stride)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(context, img, t)
            # imgs.append(img.cpu().numpy())
        img = img.permute(0, 2, 1)
        return imgs, img
    
    
    # def train_losses(self, model, x_start: torch.FloatTensor, t: torch.LongTensor):
    def get_loss(self, context, future):
        context, future = context.permute(0, 2, 1), future.permute(0, 2, 1)
        # compute train losses
        batch_size = context.size(0)
        t = torch.randint(0, self.num_steps+1, (batch_size,), device=self.device)
        noise = torch.randn_like(future)  # random noise ~ N(0, 1)
        x_noisy = self.q_sample(future, t, noise=noise)  # x_t ~ q(x_t | x_0)
        # print('diff_477', context.shape, x_noisy.shape, t.shape)
        predicted_noise = self.model(context, x_noisy, t)  # predict noise from noisy image
        loss = F.mse_loss(noise, predicted_noise)
        return loss


    def moveing_avg_filter(self, tensor, window_size):
        # padding_size = (window_size - 1) // 2
        padding_left = (window_size - 1) // 2
        padding_right = window_size // 2
        
        # padded_tensor = torch.nn.functional.pad(tensor, (padding_size, padding_size), mode='reflect')
        padded_tensor = torch.nn.functional.pad(tensor, (padding_left, padding_right), mode='reflect')
        
        unfolded = padded_tensor.unfold(dimension=2, size=window_size, step=1)
        smoothed_tensor = unfolded.mean(dim=-1)
        return smoothed_tensor


class GaussianProcess_timemixer(nn.Module):
    def __init__(self, config, device):
        super(GaussianProcess_timemixer, self).__init__()
        
        config_model = config['model']
        ctx_dim = config_model['context_dim']
        p_dim = config_model['point_dim']
        ctx_dim_out = config_model['context_dim_out']
        feat_dim = config_model['feature_dim']
        seq_length = config_model['seq_length']
        self.num_steps = config_model['diff_num_steps']
        self.stride = config_model['stride']
        
        class Config:
            def __init__(self):
                self.task_name = 'generation'
                self.seq_len = 200
                self.label_len = 0
                self.pred_len = 200
                self.down_sampling_window = 2
                self.down_sampling_layers = 3
                self.down_sampling_method = 'avg'
                self.channel_independence = 0
                self.e_layers = 3
                self.moving_avg = 5  # You can adjust this value
                self.enc_in = 6
                self.use_future_temporal_feature = 0
                self.d_model = 32
                self.embed = 'timeF'
                self.freq = 's'
                self.dropout = 0.1
                self.use_norm = 1
                self.c_out = 6
                self.num_class = 12
                self.decomp_method = 'moving_avg'
                self.d_ff = 64
                self.hidden_size = 1024

        configs = Config()
        self.model = TimeMixer(configs).to(device)
        # self.model = FilterNet(configs).to(device)
        
        self.input_dim = p_dim
        self.seq_length = seq_length
        self.device = device
        
        # beta_start, beta_end = 1e-4, 2e-2
        scale = 1000 / self.num_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        # Create a linear schedule for betas
        self.betas = torch.linspace(beta_start, beta_end, steps=self.num_steps+1).to(self.device)  # t from 0 to num_steps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a: torch.FloatTensor, t: torch.LongTensor, x_shape):
        # get the param of given timestep t
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    
    def q_sample(self, x_start: torch.FloatTensor, t: torch.LongTensor, noise=None):
        # forward diffusion (using the nice property): q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_mean_variance(self, x_start: torch.FloatTensor, t: torch.LongTensor):
        # Get the mean and variance of q(x_t | x_0).
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start: torch.FloatTensor, x_t: torch.FloatTensor, t: torch.LongTensor):
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        posterior_mean = self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_start_from_noise(self, x_t: torch.FloatTensor, t: torch.LongTensor, noise: torch.FloatTensor):
        # compute x_0 from x_t and pred noise: the reverse of `q_sample`
        return self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    """
    predict noise from start
    start is predicted by the model
    """
    def predict_noise_from_start(self, x_t: torch.FloatTensor, t: torch.LongTensor, x_start: torch.FloatTensor):
        return (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_mean_variance(self, context, x_t: torch.FloatTensor, t: torch.LongTensor, clip_denoised=True):
        # compute predicted mean and variance of p(x_{t-1} | x_t)
        # predict noise using model
        
        
        pred_noise = self.model(x_enc=x_t,
                                x_mark_enc = None,
                                x_dec = None,
                                x_mark_dec = None,
                                mask=None,
                                label=context,
                                timesteps=t)
        
        # pred_noise = self.model(context, x_t, t)
        
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1.0, max=1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, context, x_t: torch.FloatTensor, t: torch.LongTensor, clip_denoised=True):
        # denoise_step: sample x_{t-1} from x_t and pred_noise
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(context, x_t, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t).to(self.device)
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    @torch.no_grad()
    def sample_generate(self, label, num_steps=None):
        batch_size = label.shape[0]
        img = torch.randn((batch_size, self.seq_length, self.input_dim), device=self.device)
        context = label
        img = img.permute(0, 2, 1)
        imgs = []
        for i in reversed(range(0, self.num_steps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(context, img, t)
            imgs.append(img.cpu().numpy())
        img = img.permute(0, 2, 1)
        return imgs, img
    
    def get_loss_generate(self, label, future):
        future = future.permute(0, 2, 1)
        batch_size = label.shape[0]
        t = torch.randint(0, self.num_steps+1, (batch_size,), device=self.device)
        noise = torch.randn_like(future)  # random noise ~ N(0, 1)
        x_noisy = self.q_sample(future, t, noise=noise)  # x_t ~ q(x_t | x_0)
        
        predicted_noise = self.model(x_enc=x_noisy,
                                     x_mark_enc = None,
                                     x_dec = None,
                                     x_mark_dec = None,
                                     mask=None,
                                     label=label,
                                     timesteps=t)
        
        """
        for UnetGenerate
        """
        # predicted_noise = self.model(label, x_noisy, t)  # predict noise from noisy image
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    @torch.no_grad()
    # def sample(self,  image_size, batch_size=8, channels=3):
    def sample(self, context, num_steps=None):
        # denoise: reverse diffusion
        # shape = (batch_size, channels, image_size, image_size)
        # start from pure noise (for each example in the batch)
        # img = torch.randn(shape, device=self.device)  # x_T ~ N(0, 1)
        batch_size, seq_length, input_channels = context.size()
        img = torch.randn((batch_size, seq_length, input_channels), device=self.device)
        context, img = context.permute(0, 2, 1), img.permute(0, 2, 1)
        imgs = []
        # for i in tqdm(reversed(range(0, self.num_steps)), desc="sampling loop time step", total=self.num_steps):
        for i in reversed(range(0, self.num_steps, self.stride)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(context, img, t)
            # imgs.append(img.cpu().numpy())
        img = img.permute(0, 2, 1)
        return imgs, img
    
    
    # def train_losses(self, model, x_start: torch.FloatTensor, t: torch.LongTensor):
    def get_loss(self, context, future):
        context, future = context.permute(0, 2, 1), future.permute(0, 2, 1)
        # compute train losses
        batch_size = context.size(0)
        t = torch.randint(0, self.num_steps+1, (batch_size,), device=self.device)
        noise = torch.randn_like(future)  # random noise ~ N(0, 1)
        x_noisy = self.q_sample(future, t, noise=noise)  # x_t ~ q(x_t | x_0)
        # print('diff_477', context.shape, x_noisy.shape, t.shape)
        predicted_noise = self.model(context, x_noisy, t)  # predict noise from noisy image
        loss = F.mse_loss(noise, predicted_noise)
        return loss


    def moveing_avg_filter(self, tensor, window_size):
        # padding_size = (window_size - 1) // 2
        padding_left = (window_size - 1) // 2
        padding_right = window_size // 2
        
        # padded_tensor = torch.nn.functional.pad(tensor, (padding_size, padding_size), mode='reflect')
        padded_tensor = torch.nn.functional.pad(tensor, (padding_left, padding_right), mode='reflect')
        
        unfolded = padded_tensor.unfold(dimension=2, size=window_size, step=1)
        smoothed_tensor = unfolded.mean(dim=-1)
        return smoothed_tensor



# Assuming output1 and output2 have the same shape (batch_size, feature_dim)
class CombineLinear(nn.Module):
    def __init__(self, feature_dim):
        super(CombineLinear, self).__init__()
        # Linear layer for combining, outputs the same dimension
        self.linear = nn.Linear(2 * feature_dim, feature_dim)
    
    def forward(self, output1, output2):
        # pdb.set_trace()
        output1 = output1.permute(0, 2, 1)
        output2 = output2.permute(0, 2, 1)
        # Concatenate along the feature dimension
        combined = torch.cat((output1, output2), dim=2)
        # Pass through linear layer
        return self.linear(combined).permute(0, 2, 1)

class GaussianProcess_multifreq(nn.Module):
    def __init__(self, config, device):
        super(GaussianProcess_multifreq, self).__init__()
        
        config_model = config['model']
        ctx_dim = config_model['context_dim']
        p_dim = config_model['point_dim']
        ctx_dim_out = config_model['context_dim_out']
        feat_dim = config_model['feature_dim']
        seq_length = config_model['seq_length']
        self.num_steps = config_model['diff_num_steps']
        self.stride = config_model['stride']

        self.model_main = UNetGenerate(
            in_channels=p_dim,
            out_channels=p_dim
            # seq_length=seq_length
        ).to(device)
        
        self.model_res = UNetGenerate(
            in_channels=p_dim,
            out_channels=p_dim
            # seq_length=seq_length
        ).to(device)
        
        self.linear = CombineLinear(p_dim).to(device)
        
        self.input_dim = p_dim
        self.seq_length = seq_length
        self.device = device
        
        # beta_start, beta_end = 1e-4, 2e-2
        scale = 1000 / self.num_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        # Create a linear schedule for betas
        self.betas = torch.linspace(beta_start, beta_end, steps=self.num_steps+1).to(self.device)  # t from 0 to num_steps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a: torch.FloatTensor, t: torch.LongTensor, x_shape):
        # get the param of given timestep t
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    def q_sample(self, x_start: torch.FloatTensor, t: torch.LongTensor, noise=None):
        # forward diffusion (using the nice property): q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_mean_variance(self, x_start: torch.FloatTensor, t: torch.LongTensor):
        # Get the mean and variance of q(x_t | x_0).
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start: torch.FloatTensor, x_t: torch.FloatTensor, t: torch.LongTensor):
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        posterior_mean = self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_start_from_noise(self, x_t: torch.FloatTensor, t: torch.LongTensor, noise: torch.FloatTensor):
        # compute x_0 from x_t and pred noise: the reverse of `q_sample`
        return self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def predict_noise_from_start(self, x_t: torch.FloatTensor, t: torch.LongTensor, x_start: torch.FloatTensor):
        return (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_mean_variance(self, context, x_t: torch.FloatTensor, t: torch.LongTensor, clip_denoised=True, model='res'):
        if model == 'res':
            pred_noise = self.model_res(context, x_t, t)
        else:
            pred_noise = self.model_main(context, x_t, t)
        
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1.0, max=1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    # def p_mean_variance_predimg(self, context, x_t: torch.FloatTensor, t: torch.LongTensor, clip_denoised=True, model='res'):
    #     # compute predicted mean and variance of p(x_{t-1} | x_t)
    #     # predict noise using model
    #     if model == 'res':
    #         x_recon = self.model_res(context, x_t, t)
    #     else:
    #         x_recon = self.model_main(context, x_t, t)
    #     # pred_noise = self.model(context, x_t, t)
    #     # get the predicted x_0: different from the algorithm2 in the paper
    #     """
    #     different than before, because right now the model's output is actually the real img instead of the noise
    #     """
    #     # x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
    #     if clip_denoised:
    #         x_recon = torch.clamp(x_recon, min=-1.0, max=1.0)
    #     model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
    #     return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    # def p_sample_predimg(self, context, x_t: torch.FloatTensor, t: torch.LongTensor, clip_denoised=True, model='res'):
    #     # denoise_step: sample x_{t-1} from x_t and pred_noise
    #     # predict mean and variance
    #     if model == 'res':
    #         model_mean, _, model_log_variance = self.p_mean_variance_predimg(context, x_t, t, clip_denoised=clip_denoised, model='res')
    #     else:
    #         model_mean, _, model_log_variance = self.p_mean_variance_predimg(context, x_t, t, clip_denoised=clip_denoised, model='main')
            
    #     # model_mean, _, model_log_variance = self.p_mean_variance(context, x_t, t, clip_denoised=clip_denoised)
    #     noise = torch.randn_like(x_t).to(self.device)
    #     # no noise when t == 0
    #     nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
    #     # compute x_{t-1}
    #     pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    #     return pred_img

    @torch.no_grad()
    def p_sample(self, context, x_t: torch.FloatTensor, t: torch.LongTensor, clip_denoised=True, model='res'):
        # denoise_step: sample x_{t-1} from x_t and pred_noise
        # predict mean and variance
        if model == 'res':
            model_mean, _, model_log_variance = self.p_mean_variance(context, x_t, t, clip_denoised=clip_denoised, model='res')
        else:
            model_mean, _, model_log_variance = self.p_mean_variance(context, x_t, t, clip_denoised=clip_denoised, model='main')
            
        # model_mean, _, model_log_variance = self.p_mean_variance(context, x_t, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t).to(self.device)
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    
    """
    need to delete the DC components and add a hanning window before the FFT
    """
    # def norm_and_main_freq(self, x, k):
    #     # Apply FFT to get the frequency domain representation
    #     z = torch.fft.rfft(x, dim=2)
        
    #     # Find the top k largest magnitude frequency indices
    #     ks = torch.topk(z.abs(), k, dim=2)
    #     top_k_indices = ks.indices
        
    #     # Create a mask for the top k frequencies
    #     mask = torch.zeros_like(z)
    #     mask.scatter_(2, top_k_indices, 1)  # Set top k frequency indices to 1
        
    #     # Apply mask to get the top k frequency components
    #     z_m = z * mask  # z_m contains only the top k frequency components
        
    #     x_m = torch.fft.irfft(z_m, dim=2).real  # Main frequency components
    #     x_r = x - x_m  # Remaining part (without top k frequencies)
        
    #     # Get the remaining frequencies (by zeroing out the top k components)
    #     # z_r = z * (1 - mask)  # z_r contains the remaining lower magnitude frequencies
        
    #     # Apply inverse FFT to bring the signals back to time domain
    #     # x_m = torch.fft.irfft(z_m, dim=2).real  # Main frequency components
    #     # x_r = torch.fft.irfft(z_r, dim=2).real  # Remaining part (without top k frequencies)
        
    #     # Return both components
    #     return x_m, x_r


        # Remove DC component
        # x = x - x.mean(dim=2, keepdim=True)
        
        # # Apply Hanning window
        # window = torch.hann_window(x.shape[2], device=x.device)
        # x = x * window
        

    def norm_and_main_freq(self, x, k):

        # Apply FFT to get the frequency domain representation
        z = torch.fft.rfft(x, dim=2)
        
        # Find the top k largest magnitude frequency indices
        ks = torch.topk(z.abs(), k, dim=2)
        top_k_indices = ks.indices
        
        # Create a mask for the top k frequencies
        mask = torch.zeros_like(z)
        mask.scatter_(2, top_k_indices, 1)  # Set top k frequency indices to 1
        
        # Apply mask to get the top k frequency components
        z_m = z * mask  # z_m contains only the top k frequency components
        
        x_m = torch.fft.irfft(z_m, dim=2).real  # Main frequency components
        x_r = x - x_m  # Remaining part (without top k frequencies)
        
        # Return both components
        return x_m, x_r
    
    # @torch.no_grad()
    # def sample_generate_predimg(self, label, num_steps=None):
    #     batch_size = label.shape[0]
    #     img_m = torch.randn((batch_size, self.seq_length, self.input_dim), device=self.device)
    #     img_m = img_m.permute(0, 2, 1)
    #     img_r = torch.randn((batch_size, self.seq_length, self.input_dim), device=self.device)
    #     img_r = img_r.permute(0, 2, 1)
    #     context = label
    #     # img = img.permute(0, 2, 1)
    #     imgs = []
    #     for i in reversed(range(0, self.num_steps)):
    #         t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
    #         img_m = self.p_sample_predimg(context, img_m, t, model='main')
    #         img_r = self.p_sample_predimg(context, img_r, t, model='res')
            
            
    #         img = self.linear(img_r, img_m)
            
    #         imgs.append(img.cpu().numpy())
    #     img = img.permute(0, 2, 1)
    #     return imgs, img
    
    @torch.no_grad()
    def sample_generate_prednoise(self, label, num_steps=None):
        batch_size = label.shape[0]
        img_m = torch.randn((batch_size, self.seq_length, self.input_dim), device=self.device)
        img_m = img_m.permute(0, 2, 1)
        img_r = torch.randn((batch_size, self.seq_length, self.input_dim), device=self.device)
        img_r = img_r.permute(0, 2, 1)
        context = label
        # img = img.permute(0, 2, 1)
        imgs = []
        for i in reversed(range(0, self.num_steps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img_m = self.p_sample(context, img_m, t, model='main')
            img_r = self.p_sample(context, img_r, t, model='res')
            img = img_m + img_r
            imgs.append(img.cpu().numpy())
        img = img.permute(0, 2, 1)
        return imgs, img
    
    
    # """
    # change the loss function to use the combined output
    # change the model's output to be the real img instead of the noise
    # """
    # def get_loss_generate_predimg(self, label, future):
    #     future = future.permute(0, 2, 1)
    #     future_m, future_r = self.norm_and_main_freq(future, k=3)
    #     batch_size = label.shape[0]
    #     t = torch.randint(0, self.num_steps+1, (batch_size,), device=self.device)
        
    #     noise_r = torch.randn_like(future_r)  # random noise ~ N(0, 1)
    #     noise_m = torch.randn_like(future_m)
        
    #     x_noisy_r = self.q_sample(future_r, t, noise=noise_r)
    #     x_noisy_m = self.q_sample(future_m, t, noise=noise_m)
        
    #     predict_img_r = self.model_res(label, x_noisy_r, t)
    #     predict_img_m = self.model_main(label, x_noisy_m, t)
        
    #     predict_img = self.linear(predict_img_r, predict_img_m)
    #     loss = F.mse_loss(predict_img, future)
    #     return loss
        
        # loss_r = F.mse_loss(noise_r, predicted_noise_r)
        # loss_m = F.mse_loss(noise_m, predicted_noise_m)
        # loss = alpha * loss_r + (1 - alpha) * loss_m
        
        # this part should be, first get the final output, then compute the total loss
        # like
        # img = self.linear(predict_img_r, predict_img_m)
        # loss = F.mse_loss(img, future)
        
        # return loss, loss_r, loss_m
        
    def get_loss_generate_prednoise(self, label, future):
        alpha = 0.6
        future = future.permute(0, 2, 1)
        # pdb.set_trace()
        future_m, future_r = self.norm_and_main_freq(future, k=3)
        batch_size = label.shape[0]
        t = torch.randint(0, self.num_steps+1, (batch_size,), device=self.device)
        
        noise_r = torch.randn_like(future_r)  # random noise ~ N(0, 1)
        noise_m = torch.randn_like(future_m)
        
        x_noisy_r = self.q_sample(future_r, t, noise=noise_r)
        x_noisy_m = self.q_sample(future_m, t, noise=noise_m)
        
        predict_noise_r = self.model_res(label, x_noisy_r, t)
        predict_noise_m = self.model_main(label, x_noisy_m, t)
        
        loss = alpha * F.mse_loss(noise_r, predict_noise_r) + (1-alpha) * F.mse_loss(noise_m, predict_noise_m)
        
        return loss

        