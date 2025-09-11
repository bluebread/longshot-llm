import math
from itertools import combinations
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel
from transformers import GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions
)


class GPT2ForLongshotConfig(PretrainedConfig):
    """
    """
    
    def __init__(
        self, 
        num_vars: int,
        width: int,
        n_embed_lit: int,
        ub_q: float,
        alpha: float,
        beta: float,
        gamma: float,
        share_semantic: bool,
        universal: bool,
        gpt2_config: GPT2Config
    ):
        """
        """
        self.num_vars = num_vars
        self.width = width
        self.n_embed_lit = n_embed_lit
        self.ub_q = ub_q
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.share_semantic = share_semantic
        self.universal = universal
        self.gpt2_config = gpt2_config
        
        self.n_token_char = 5 if share_semantic else 3 * num_vars + 2
        self.vocab_size = 2 * math.comb(num_vars, width) * (2 ** width)
        self.n_embed_gate = self.n_embed_lit * (self.num_vars + 1)
        self.dim_model_output = 2 * num_vars + 2
        
        super().__init__()
        

class GPT2ForLongshot(GPT2PreTrainedModel, GenerationMixin):
    """
    """
    
    def __init__(self, config: GPT2ForLongshotConfig):
        """
        """
        super().__init__(config.gpt2_config)
        
        # 0,1 for ADD/DEL types, 2-4 for positive/negative/omitted literals
        self.cofnig = config
        self.gpt2_config = config.gpt2_config
        self.num_vars = config.num_vars
        self.width = config.width
        self.n_embed_lit = config.n_embed_lit
        self.vocab_size = config.vocab_size
        self.n_embed_gate = config.n_embed_gate
        self.ub_q = config.ub_q
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.share_semantic = config.share_semantic
        self.universal = config.universal
        
        self.embedding = torch.nn.Embedding(config.n_token_char, config.n_embed_lit)
        self.proj1 = torch.nn.Linear(self.n_embed_gate, config.gpt2_config.n_embd)
        self.model = GPT2Model(config.gpt2_config)
        # TODO: change proj2 from single Linear layer to MLP
        self.proj2 = torch.nn.Linear(config.gpt2_config.n_embd, config.dim_model_output)
        
        self.init_weights()
    
        # Remove positional encoding (PE) from GPT2Model
        self.model.wpe = torch.nn.Embedding(
            config.gpt2_config.n_positions, 
            config.gpt2_config.n_embd
        )
        with torch.no_grad():
            self.model.wpe.weight.fill_(0)
            self.model.wpe.weight.requires_grad = False
    
    
    def _decode_input_to_embedding(self, input_ids: torch.Tensor) -> tuple[torch.Tensor]:
        x = torch.abs(input_ids)
        concat_later = [(input_ids < 0).int()] # token type
        base_fn = lambda i: 2 if self.share_semantic else 3 * i + 2
        
        for i in range(self.num_vars):
            z = (((x >> i) & 0x1) | ((x >> (i + 32 - 1)) & 0x2))
            concat_later.append(z)
        
        concat_embd_idx = [
            self.embedding(c + base_fn(i)) 
            for i, c in enumerate(concat_later)
        ]
        
        decoded_input = torch.cat(concat_later, dim=-1)
        embd = torch.cat(concat_embd_idx, dim=-1)
        
        return embd, decoded_input
    

    def _sum_over_subsets(self, values: torch.Tensor, complement: bool) -> torch.Tensor:
        """
        Parallel computation of subset sums via bit manipulation.
        
        Efficiently computes sums for all 2^k possible subsets of k elements
        using vectorized operations. Each subset is represented by a bitmask.
        
        Algorithm:
        - Creates 2^k x k binary matrix where row i represents subset mask i
        - Matrix multiplication computes all subset sums in parallel
        
        Args:
            values: Tensor (..., k) with values to aggregate.
        
        Returns:
            Tensor (..., 2^k) where position [..., mask] contains sum of
            values at indices where mask has bit set. Example: mask=5 (0b101)
            yields values[...,0] + values[...,2].
        """
        k = values.size(-1)
        masks = torch.arange(1 << k, device=values.device).unsqueeze(1)
        masks = masks.bitwise_and(1 << torch.arange(k, device=values.device))

        # TODO: need to check
        if complement:
            masks = masks.eq(0).float()
        else:
            masks = masks.ne(0).float()

        return values @ masks.transpose(0, 1)
    
    
    def _sum_over_combinations(self, values: torch.Tensor) -> torch.Tensor:
        """
        """
        di = math.comb(self.num_vars, self.width)
        ss = torch.zeros((di, self.num_vars), dtype=torch.uint8)
        
        for i, c in enumerate(combinations(range(self.num_vars), self.width)):
            ss[i, list(c)] = 1
        
        return values @ ss.transpose(0, 1)
    
    
    def _gather_combinations(self, values: torch.Tensor) -> torch.Tensor:
        """
        """    
        subsets = list(combinations(range(self.num_vars), self.width))
        ss = torch.tensor(subsets, device=self.device, dtype=torch.long)
        batch_shape = values.shape[:-1]
        
        for _ in batch_shape:
            ss.unsqueeze(0)
        ss = ss.expand(*batch_shape, -1, -1)
        
        values = values.unsqueeze(-2)
        values = values.expand(*((-1,) * len(batch_shape)), len(subsets), -1)
        
        return values.gather(dim=-1, index=ss)

    
    def _calculate_avgQ_loss(
        self, 
        q: torch.Tensor,
        labels: torch.Tensor, 
        attn: torch.Tensor
    ) -> torch.Tensor | None:
        """
        """
        q = q.squeeze(-1) * attn
        labels = labels * attn
        D = (self.ub_q - labels)
        D_hat = (self.ub_q - q)
        Q = labels
        Q_hat = q
        
        l1 = F.mse_loss(Q_hat, Q, reduction='mean')
        l2 = F.mse_loss(torch.exp(- D_hat), torch.exp(- D), reduction='mean')
        
        return self.alpha * l1 + self.beta * l2
        
        
    def _calculate_token_loss(
        self, 
        zeta: torch.Tensor, # [batch, seq, 1]
        phi: torch.Tensor, # [batch, seq, n]
        psi: torch.Tensor, # [batch, seq, n]
        decoded_input: torch.Tensor, # [batch, seq, n + 1]
        attn: torch.Tensor # [batch, seq]
    ) -> torch.Tensor | None:
        """
        """
        nxt_tokens = decoded_input[..., 0:, :] # [batch, seq, n + 1]
        tt = nxt_tokens[..., 0] # [batch, seq - 1, 1]
        x = nxt_tokens[..., 1:] # [batch, seq - 1, n]
        idx = torch.topk(x, self.width, dim=-1).indices # [batch, seq - 1, w]
        sgn = x.eq(2).int() # [batch, seq - 1, w]
        
        ze = zeta[..., :-1, :] # [batch, seq - 1, 1]
        ph = phi[..., :-1, :].gather(dim=-1, index=idx) # [batch, seq - 1, w]
        ps = psi[..., :-1, :].gather(dim=-1, index=idx) # [batch, seq - 1, w]
        tb = Bernoulli(logits=ze)
        sb = Bernoulli(logits=ps)
        
        tp = tb.log_prob(tt).squeeze(-1)
        ip = ph.sum(dim=-1)
        sp = sb.log_prob(sgn).sum(dim=-1)
        logp = (tp + ip + sp) * attn[..., 0:]
        
        return - logp.mean()
        
    def _calculate_logits(
        self, 
        zeta: torch.Tensor, 
        phi: torch.Tensor, 
        psi: torch.Tensor,
        slice_indices: torch.Tensor | slice,
    ) -> torch.Tensor:
        """
        """
        n = self.num_vars
        w = self.width
        dt = 2
        di = math.comb(n, w)
        ds = 2 ** w
        d = dt * di * ds
        assert d == self.vocab_size, "logits size doesn't match the size of the vocabulary"
        
        ze = zeta[..., slice_indices, :]
        ph = phi[..., slice_indices, :]
        ps = psi[..., slice_indices, :]
        batch_shape = ph.shape[:-1]
        
        add_p = - F.softplus(ze)
        del_p = torch.log(1 - 1 / (1 + torch.exp(ze)))
        logp_t = torch.cat([add_p, del_p], dim=-1) # [batch, seq, 2]

        logp_i = self._sum_over_combinations(ph) # [batch, seq, C(n,w)]
        
        pp = self._gather_combinations(- F.softplus(ps))
        np = self._gather_combinations(torch.log(1 - 1 / (1 + torch.exp(ps))))
        pp = self._sum_over_subsets(pp, complement=False)
        np = self._sum_over_subsets(np, complement=True)
        logp_s = pp + np # [batch, seq, C(n,w), 2**w]
        
        logp_t = logp_t[..., :, None, None] # [batch, seq, 2, 1, 1]
        logp_i = logp_i[..., None, :, None] # [batch, seq, 1, C(n,w), 1]
        logp_s = logp_s[..., None, :, :] # [batch, seq, 1, C(n,w), 2**w]
        
        return (logp_t + logp_i + logp_s).view(*batch_shape, d)
        
    
    def forward(
        self, 
        input_ids: torch.LongTensor = None,
        past_key_values: tuple[tuple[torch.Tensor]] = None,
        attention_mask: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        logits_to_keep: int | torch.Tensor = 0,
    ) -> tuple[torch.Tensor, ...] | CausalLMOutputWithPast:
        """
        """
        n = self.num_vars
        attn = attention_mask
        
        slice_indices = logits_to_keep
        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None)
            
        x, decoded_input = self._decode_input_to_embedding(input_ids)
        x = self.proj1(x)
        transformer_outputs: BaseModelOutputWithPastAndCrossAttentions = self.model(
            inputs_embeds=x, 
            past_key_values=past_key_values,
            attention_mask=attention_mask, 
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        past_key_values = transformer_outputs.past_key_values if use_cache else None
        hidden_state = transformer_outputs.last_hidden_state
        attentions = transformer_outputs.attentions
        y = self.proj2(hidden_state)
        zeta, phi, psi, q = torch.split(y, [1, n, n, 1], dim=-1)
        
        loss = None
        if labels is not None:
            loss_avgQ = self._calculate_avgQ_loss(q, labels, attn)
            loss_token = self._calculate_token_loss(zeta, phi, psi, decoded_input, attn)
            loss = loss_avgQ + loss_token
            
        logits = self._calculate_logits(zeta, phi, psi, slice_indices)
    
        if not return_dict:
            return (
                v for v in [loss, logits, past_key_values, hidden_state, attentions]
                if v is not None
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_state,
            attentions=attentions,
        )