import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.amp import autocast
import einx

# --- Helper Functions ---

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))

def down(x): return F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

def up(x): return F.interpolate(x, scale_factor=(1, 2.0, 2.0), mode='nearest')


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)


class TimestepEmbedder(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = (t[:, None].float() * 1000.0) * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)

class AdaLNModulation(Module):
    def __init__(self, dim, out_multiplier=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, out_multiplier * dim, bias=True)
        )
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, c):
        return self.net(c)

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SpatioTemporalAxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10, dim_t=None, dim_spatial=None):
        super().__init__()
        self.dim = dim

        if dim_t is None or dim_spatial is None:
            dim_t = dim // 3
            dim_t -= (dim_t % 2)  
            dim_spatial = (dim - dim_t) // 2
            dim_spatial -= (dim_spatial % 2)

        self.register_buffer('scales_t', torch.linspace(1., max_freq / 2, dim_t // 2))
        self.register_buffer('scales_spatial', torch.linspace(1., max_freq / 2, dim_spatial // 2))

    @autocast(device_type='cuda', enabled=False)
    def forward(self, device, dtype, t, n):
        seq_t = torch.linspace(-1., 1., steps=t, device=device, dtype=dtype).unsqueeze(-1)
        seq_spatial = torch.linspace(-1., 1., steps=n, device=device, dtype=dtype).unsqueeze(-1)

        seq_t = seq_t * self.scales_t.to(dtype) * math.pi
        seq_spatial = seq_spatial * self.scales_spatial.to(dtype) * math.pi

        t_sinu = repeat(seq_t, 't d -> t h w d', h=n, w=n)
        h_sinu = repeat(seq_spatial, 'h d -> t h w d', t=t, w=n) 
        w_sinu = repeat(seq_spatial, 'w d -> t h w d', t=t, h=n) 

        sin_cat = torch.cat((t_sinu.sin(), h_sinu.sin(), w_sinu.sin()), dim=-1)
        cos_cat = torch.cat((t_sinu.cos(), h_sinu.cos(), w_sinu.cos()), dim=-1)

        sin = rearrange(sin_cat, 't h w d -> (t h w) d')
        cos = rearrange(cos_cat, 't h w d -> (t h w) d')

        sin = repeat(sin, 'seq d -> () () seq (j d)', j=2)
        cos = repeat(cos, 'seq d -> () () seq (j d)', j=2)
        
        return sin, cos

class MultiScaleJointAttention(nn.Module):
    def __init__(self, dim, num_scales, context_dim=4096, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.num_scales = num_scales
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
   
        self.to_qkv = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 3 * inner_dim, bias=False),
                Rearrange('b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
            )
            for _ in range(num_scales)
        ])
        self.to_out_vid = nn.ModuleList([
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            for _ in range(num_scales)
        ])
       
        self.to_qkv_txt = nn.Sequential(
            nn.Linear(context_dim, 3 * inner_dim, bias=False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        )
        self.to_out_txt = nn.Sequential(nn.Linear(inner_dim, context_dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, xs, context, sincos_list):
        device = xs[0].device
        text_len = context.shape[1]
        lens = [x.shape[1] for x in xs] 

        # Project text
        qkv_txt = self.to_qkv_txt(context)
        q_txt, k_txt, v_txt = qkv_txt[0], qkv_txt[1], qkv_txt[2]

        # Project video 
        qkvs = [qkv_layer(x) for qkv_layer, x in zip(self.to_qkv, xs)]
        qs, ks, vs = zip(*qkvs)
        
        # Apply RoPE to Video Queries and Keys at each scale before concatenation
        qs_rotated = []
        ks_rotated = []
        for q, k, (sin, cos) in zip(qs, ks, sincos_list):
            qs_rotated.append(apply_rotary_emb(q, sin, cos))
            ks_rotated.append(apply_rotary_emb(k, sin, cos))

        # concat video scales 
        q_vid = torch.cat(qs_rotated, dim=2)
        k_vid = torch.cat(ks_rotated, dim=2)
        v_vid = torch.cat(vs, dim=2)

        # concat for joint attention
        q = torch.cat([q_txt, q_vid], dim=2)
        k = torch.cat([k_txt, k_vid], dim=2)
        v = torch.cat([v_txt, v_vid], dim=2)

        # build the mask
        scale_indices = torch.arange(len(xs), device=device)
        lens_t_tensor = torch.tensor(lens, device=device)

        token_scales = torch.repeat_interleave(scale_indices, lens_t_tensor)
        q_scales = rearrange(token_scales, 'q -> q 1')
        k_scales = rearrange(token_scales, 'k -> 1 k')

        video_causal_mask = k_scales > q_scales 
        
        total_len = text_len + sum(lens)
        joint_mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)
        joint_mask[text_len:, text_len:] = video_causal_mask

        # global attention
        sim = torch.einsum('b h q d, b h k d -> b h q k', q, k) * self.scale
        sim.masked_fill_(joint_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out_global = torch.einsum('b h q k, b h k d -> b h q d', attn, v)
        out_global = rearrange(out_global, 'b h n d -> b n (h d)')

        out_txt = out_global[:, :text_len, :]
        txt_out = self.to_out_txt(out_txt)

        out_vid = out_global[:, text_len:, :]
        outs_vid = torch.split(out_vid, lens, dim=1)
        
        vid_outs = [self.to_out_vid[i](outs_vid[i]) for i in range(len(xs))]

        return vid_outs, txt_out

class DiTBlock(Module):
    def __init__(self, dim, num_scales, heads, dim_head, mlp_dim, context_dim=4096, dropout=0.):
        super().__init__()
        self.num_scales = num_scales

        self.norm1_vid = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(num_scales)])
        self.norm2_vid = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(num_scales)])
        self.ff_vid = nn.ModuleList([FeedForward(dim, mlp_dim, dropout) for _ in range(num_scales)])
        self.adaln_vid = nn.ModuleList([AdaLNModulation(dim, out_multiplier=6) for _ in range(num_scales)])
    
        self.norm1_txt = nn.LayerNorm(context_dim)
        self.norm2_txt = nn.LayerNorm(context_dim)
        self.ff_txt = FeedForward(context_dim, context_dim * 4, dropout) 

        self.joint_attn = MultiScaleJointAttention(dim, num_scales, context_dim, heads, dim_head, dropout)

    def forward(self, xs, c, context, sincos_list):
        chunks = [adaln(c).chunk(6, dim=-1) for adaln in self.adaln_vid]
        msa_inputs = [modulate(norm(x), ch[0], ch[1]) for x, norm, ch in zip(xs, self.norm1_vid, chunks)]

        attn_outs, context_outs = self.joint_attn(msa_inputs, self.norm1_txt(context), sincos_list)

        context = context + context_outs
        context = context + self.ff_txt(self.norm2_txt(context))

        outs = []
        for x, attn_out, norm2_vid, ff_vid, (_, _, gate_msa, shift_mlp, scale_mlp, gate_mlp) in zip(
            xs, attn_outs, self.norm2_vid, self.ff_vid, chunks
        ):
            x = x + gate_msa.unsqueeze(1) * attn_out
            x = x + gate_mlp.unsqueeze(1) * ff_vid(modulate(norm2_vid(x), shift_mlp, scale_mlp))
            outs.append(x)

        return outs, context

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels):
        labels = labels.long()
        labels = torch.where(labels < 0, self.num_classes, labels)
        return self.embedding_table(labels)

class LapFlowDiT(Module):
    def __init__(
        self,
        base_image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        channels = 3,
        dim_head = 64,
        num_scales = 3,
        dropout = 0.,
        accept_cond = False,
        dim_cond = None,
        cond_as_labels = False,
        num_classes = None,
        sinusoidal_pos_emb_theta = 10000
    ):
        super().__init__()
        self.num_scales = num_scales
        patch_dim = channels * patch_size * patch_size

        grids = [(base_image_size // (2 ** i)) // patch_size for i in reversed(range(num_scales))]

        def linear():
            l = nn.Linear(dim, patch_dim, bias=True)
            nn.init.constant_(l.weight, 0)
            nn.init.constant_(l.bias, 0)
            return l

        self.patch_embeds = nn.ModuleList([
            nn.Sequential(
                Rearrange('b c t (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            ) for _ in range(num_scales)
        ])

        # RoPE operates on dim_head, so we instantiate it for each scale
        self.pos_embeds = nn.ModuleList([
            SpatioTemporalAxialRotaryEmbedding(dim=dim_head) for _ in range(num_scales)
        ])

        self.final_adalns = nn.ModuleList([AdaLNModulation(dim, out_multiplier=2) for _ in range(num_scales)])
        self.final_norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(num_scales)])
        self.final_linears = nn.ModuleList([linear() for _ in range(num_scales)])

        self.unpatchifys = nn.ModuleList([
            Rearrange('b (t h w) (p1 p2 c) -> b c t (h p1) (w p2)', h=g, w=g, p1=patch_size, p2=patch_size)
            for g in grids
        ])

        self.dropout = nn.Dropout(dropout)
        self.t_embedder = TimestepEmbedder(dim)

        context_dim = dim_cond if dim_cond is not None else dim

        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                num_scales=num_scales,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                context_dim=context_dim,
                dropout=dropout
            )
            for _ in range(depth)
        ])

    def forward(self, imgs_list, times, cond=None):
        xs = []
        sincos_list = []
        
        for img, patch_embed, pos_embed in zip(imgs_list, self.patch_embeds, self.pos_embeds):
            x = patch_embed(img) # Shape: [b, t, n, d] where n = h*w
            
            # Extract dynamically to build RoPE frequencies
            b, t, n, _ = x.shape
            grid_size = int(math.sqrt(n)) 
            
            # Generate Sin/Cos frequencies for this scale
            sin, cos = pos_embed(x.device, x.dtype, t, grid_size)
            sincos_list.append((sin, cos))
            
            x = rearrange(x, 'b t n d -> b (t n) d')
            xs.append(self.dropout(x))

        assert exists(times), "Time embedding 't' or 'times' must be provided to LapFlowDiT"
        c = self.t_embedder(times)

        context = cond 
        if context is not None and context.ndim == 2:
            context = context.unsqueeze(1)

        for block in self.blocks:
            xs, context = block(xs, c, context, sincos_list)

        outs = []
        for x, adaln, norm, linear, unpatch in zip(
            xs, self.final_adalns, self.final_norms, self.final_linears, self.unpatchifys
        ):
            shift, scale = adaln(c).chunk(2, dim=-1)
            x = modulate(norm(x), shift, scale)
            x = linear(x)
            outs.append(unpatch(x))

        return outs


class LapFlow(Module):
    def __init__(
        self,
        model: Module,
        critical_times=None,
        loss_weights=None,
        times_cond_kwarg='times',
        data_shape=None,
        normalize_data_fn=lambda t: t,
        unnormalize_data_fn=lambda t: t,
        cfg_scale=1.0,
        vae: Module = None,
        vae_scale_factor: float = 1.0
    ):
        super().__init__()

        self.model = model
        self.num_scales = model.num_scales
        self.times_cond_kwarg = times_cond_kwarg
        self.data_shape = data_shape
        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn
        self.cfg_scale = cfg_scale
        self.vae = vae
        self.vae_scale_factor = vae_scale_factor

        assert self.num_scales in [2, 3]

        if critical_times is None:
            if self.num_scales == 2:
                critical_times = [0.0, 0.5]
            elif self.num_scales == 3:
                critical_times = [0.0, 0.33, 0.67]

        self.register_buffer('critical_times', torch.tensor(critical_times, dtype=torch.float32))

        weights = default(loss_weights, [1.0] * self.num_scales)
        self.register_buffer('loss_weights', torch.tensor(weights, dtype=torch.float32))


    def get_laplacian_pyramid(self, x):

        if self.num_scales == 2:
            coarse = down(x)
            return [coarse, x - up(coarse)]

        elif self.num_scales == 3:
            low = down(x)
            coarse = down(low)
            return [coarse, low - up(coarse), x - up(low)]

    @torch.no_grad()
    def sample(self, batch_size=1, data_shape=None, steps=30, **kwargs):
        if exists(self.vae) and exists(self.data_shape):
            data_shape = self.data_shape
        else:
            data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape)

        device = next(self.model.parameters()).device

        noise = torch.randn((batch_size, *data_shape), device=device)
        noise_pyramid = self.get_laplacian_pyramid(noise)

        time_points = torch.cat([self.critical_times, torch.tensor([1.0], device=device)])

        t_starts = time_points[:-1]
        t_ends = time_points[1:]
        durations = t_ends - t_starts

        steps = torch.clamp((steps * durations).int(), min=1)

        pyd_states = [
            noise * (1.0 - timer_th.item())
            for noise, timer_th in zip(noise_pyramid, self.critical_times)
        ]

        for i in range(self.num_scales):
            t_start = t_starts[i]
            t_end = t_ends[i]
            step_count = steps[i]
            dt = (t_end - t_start) / step_count

            times = torch.linspace(t_start, t_end, step_count + 1, device=device)
            
            # 1 for coarse, 2 for mid, 3 for fine
            active_count = i + 1 

            for time in times[:-1]:
                time_val = time.item()
                time_tensor = repeat(torch.tensor([time_val], device=device), '1 -> b', b=batch_size)
 
                time_kwarg = {self.times_cond_kwarg: time_tensor} if exists(self.times_cond_kwarg) else dict()

                # only pass the active states to the model
                model_inputs = pyd_states[:active_count]

                if 'cond' in kwargs and self.cfg_scale > 1.0:
                    cond = kwargs['cond']
                    preds_cond = self.model(model_inputs, **time_kwarg, **kwargs)

                    kwargs['cond'] = torch.full_like(cond, -1)
                    preds_uncond = self.model(model_inputs, **time_kwarg, **kwargs)
                    kwargs['cond'] = cond

                    preds = [
                        pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)
                        for pred_cond, pred_uncond in zip(preds_cond, preds_uncond)
                    ]
                else:
                    preds = self.model(model_inputs, **time_kwarg, **kwargs)

                for j, pred in enumerate(preds):
                    pyd_states[j] = pyd_states[j] + pred * dt

        curr = pyd_states[0]
        for i in range(1, self.num_scales):
            curr = up(curr) + pyd_states[i]

        if exists(self.vae):
            curr = curr / self.vae_scale_factor
            decoded = self.vae.decode(curr)
            curr = decoded.sample

        curr = self.unnormalize_data_fn(curr)
        return curr.clamp(0., 1.)


    def forward(self, data, **kwargs):
        if isinstance(data, (tuple, list)):
            actual_image, cond = data[0], data[1]
            cond = rearrange(cond, 'b 1 -> b') if cond.ndim == 2 and cond.shape[1] == 1 else cond

            if self.training and torch.rand(1).item() < 0.1:
                cond = torch.full_like(cond, -1)

            kwargs['cond'] = cond
            data = actual_image


        data = self.normalize_data_fn(data)

        if exists(self.vae):
            with torch.no_grad():
                self.vae.eval()
                encoded = self.vae.encode(data)
                data = encoded.latent_dist.sample()
                data = data * self.vae_scale_factor

        shape, ndim = data.shape, data.ndim

        self.data_shape = default(self.data_shape, shape[1:])
        batch, device = shape[0], data.device

        data_list = self.get_laplacian_pyramid(data)
        noise_list = self.get_laplacian_pyramid(torch.randn_like(data))

        active_scale = torch.randint(0, self.num_scales, (1,)).item()

        start_time = self.critical_times[active_scale]

        times = torch.lerp(start_time, torch.tensor(1.0, device=device), torch.rand(batch, device=device))

        alphas = torch.clamp((rearrange(times, 'b -> b 1') - self.critical_times) / (1 - self.critical_times), min=0.0)
        sigma = append_dims(1.0 - times, data.ndim - 1)

        noised_list = [
            (append_dims(alpha, data.ndim - 1) * data) + (sigma * noise)
            for alpha, data, noise in zip(alphas.unbind(dim=1), data_list, noise_list)
        ]

        target_velocities = [
            (1.0 / (1 - timer_th)) * data - noise
            for timer_th, data, noise in zip(self.critical_times, data_list, noise_list)
        ]

        time_kwarg = {self.times_cond_kwarg: times} if exists(self.times_cond_kwarg) else dict()

        preds_list = self.model(noised_list[:active_scale + 1], **time_kwarg, **kwargs)

        total_loss = 0.0

        for pred, target, weight in zip(preds_list, target_velocities, self.loss_weights):
            total_loss += F.mse_loss(pred, target) * weight

        return total_loss