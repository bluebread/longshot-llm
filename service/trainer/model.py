import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel
from distribution import GateTokenDistribution

class GPT2ForLongshot(GPT2PreTrainedModel):
    def __init__(
        self, 
        num_vars: int,
        width: int,
        n_embed_lit: int,
        ub_q: float,
        alpha: float,
        beta: float,
        gamma: float,
        config: GPT2Config
    ):
        """
        """
        super().__init__(config)
        
        # 0,1 for ADD/DEL types, 2-4 for positive/negative/omitted literals
        self.num_vars = num_vars
        self.width = width
        self.n_embed_lit = n_embed_lit
        self.vocab_size = 3 * num_vars + 2
        self.n_embed_gate = n_embed_lit * (num_vars + 1)
        self.ub_q = ub_q
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.embedding = torch.nn.Embedding(self.vocab_size, n_embed_lit)
        self.proj1 = torch.nn.Linear(self.n_embed_gate, config.n_embd)
        self.model = GPT2Model(config)
        self.proj2 = torch.nn.Linear(config.n_embd, 2 * num_vars + 2)
        
        self.init_weights()
    
    
    def loss_function(
        self, 
        y: torch.Tensor, 
        input_ids: torch.Tensor, 
        labels: torch.Tensor, 
        attn: torch.Tensor
    ):
        """
        """
        param, q = torch.split(y, [2 * self.num_vars + 1, 1], dim=-1)
        dist = GateTokenDistribution(param, self.width)
        nxt_ids = input_ids[..., 0:]
        
        # The loss of predicting next tokens
        tt = (nxt_ids < 0).int().unsqueeze(-1) # token type
        s = torch.abs(nxt_ids)
        x = torch.cat([
            (((s >> i) & 0x1) | ((s >> (i + 32 - 1)) & 0x2)).unsqueeze(-1)
            for i in range(self.num_vars)
        ], dim=-1) # encoded literals
        idx = torch.topk(x, self.width, dim=-1).indices
        sgn = (s.unsqueeze(-1) & (1 << (idx + 32))).gt(0).int()
        nxt = torch.cat([tt, idx, sgn], dim=-1) # next tokens
        loss_nxt = (- dist.log_prob(nxt) * attn[..., 0:]).sum()
        
        # The loss of avgQ
        q = q.squeeze(-1) * attn
        labels = labels * attn
        d = (self.ub_q - labels)
        d_hat = (self.ub_q - q)
        q = labels
        q_hat = q
        
        l1 = F.mse_loss(torch.exp(- d_hat), torch.exp(- d), reduction='mean')
        l2 = F.mse_loss(q_hat, q, reduction='mean')
        loss_avgQ = self.alpha * l1 + self.beta * l2
        
        return loss_avgQ + self.gamma * loss_nxt

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        attention_mask: torch.LongTensor = None, 
        labels: torch.FloatTensor = None,
    ):
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Optional tensor of shape (batch_size, seq_len)
        
        Returns:
            Tensor of shape (batch_size, seq_len) with predicted Q-values
        """
        # Embed input ids
        x = torch.abs(input_ids)
        concat_later = [(input_ids < 0).int()]
        
        for i in range(self.num_vars):
            base = 3 * i + 2
            z = (((x >> i) & 0x1) | ((x >> (i + 32 - 1)) & 0x2)) + base
            concat_later.append(z)
            assert torch.all((z >= base) & (z < base + 3)), "Literal encoding out of range"
        
        x = torch.cat([self.embedding(c) for c in concat_later], dim=-1)
        x = self.proj1(x)
        x, = self.model(
            inputs_embeds=x, 
            attention_mask=attention_mask, 
            use_cache=False,
            return_dict=False
        )
        y = self.proj2(x)
        
        loss = None
        if labels is not None:
            attn = attention_mask
            loss = self.loss_function(y, input_ids, labels, attn).unsqueeze(0)
            
        return (loss, y) if loss is not None else y
        

if __name__ == "__main__":
    # Example usage
    from dataset import TrajectoryDataset
    from collator import TrajectoryCollator
    
    n = 3
    k = 2
    
    dataset = TrajectoryDataset(num_vars=n, width=k)
    collector = TrajectoryCollator(num_vars=n)
    sample_batch = [dataset[i] for i in range(4)]  # Get 4 samples
    batch = collector(sample_batch)
    
    from transformers import set_seed
    set_seed(42)
    
    model_config = GPT2Config(
        vocab_size=1,  # Not used since we provide embeddings directly
        n_positions=64,
        n_embd=64,
        n_layer=4,
        n_head=4,
    )
    
    model = GPT2ForLongshot(
        num_vars=n,
        width=k,
        n_embed_lit=16,
        ub_q=3.0,
        alpha=0.0,
        beta=0.0,
        gamma=1.0,
        config=model_config
    )
    
    loss, logits = model(**batch)
    
    print("Embedding Dimension (gate): ", model.n_embed_gate)
    print("Loss: ", loss)
    print(f"Logits {logits.shape}: \n", logits)