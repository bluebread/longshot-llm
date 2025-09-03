import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class GPT2ForLongshot(GPT2PreTrainedModel):
    def __init__(
        self, 
        num_vars: int,
        n_embed_lit: int,
        ub_q: float,
        alpha: float,
        beta: float,
        config: GPT2Config
    ):
        """
        """
        super().__init__(config)
        
        # 0,1 for ADD/DEL types, 2-4 for positive/negative/omitted literals
        self.num_vars = num_vars
        self.n_embed_lit = n_embed_lit
        self.vocab_size = 3 * num_vars + 2
        self.n_embed_gate = n_embed_lit * (num_vars + 1)
        self.ub_q = ub_q
        self.alpha = alpha
        self.beta = beta
        
        self.embedding = torch.nn.Embedding(self.vocab_size, n_embed_lit)
        self.proj1 = torch.nn.Linear(self.n_embed_gate, config.n_embd)
        self.model = GPT2Model(config)
        self.proj2 = torch.nn.Linear(config.n_embd, 1)
        
        self.init_weights()
    
    
    def loss_function(self, y, labels):
        """
        """
        d = (self.ub_q - labels)
        d_hat = (self.ub_q - y)
        q = labels
        q_hat = y
        
        l1 = F.mse_loss(torch.exp(- d_hat), torch.exp(- d))
        l2 = F.mse_loss(q_hat, q)
        loss = self.alpha * l1 + self.beta * l2
        
        return loss

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        attention_mask: torch.LongTensor = None, 
        labels: torch.FloatTensor = None,
        retrun_dict: bool = False
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
        y = self.proj2(x).squeeze(-1)
        
        loss = None
        if labels is not None:
            attn = attention_mask
            loss = self.loss_function(y * attn, labels * attn)
            
        if not retrun_dict:
            return (loss, y) if loss is not None else y
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=y,
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )
        

if __name__ == "__main__":
    # Example usage
    from dataset import TrajectoryDataset
    from collector import TrajectoryCollector
    
    dataset = TrajectoryDataset(num_vars=3, width=2)
    collector = TrajectoryCollector()
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
        num_vars=3,
        n_embed_lit=16,
        ub_q=3.0,
        alpha=0.9,
        beta=0.1,
        config=model_config
    )
    
    loss, logits = model(**batch)
    
    print("Embedding Dimension (gate): ", model.n_embed_gate)
    print("Loss: ", loss)
    print("Logits: \n", logits)