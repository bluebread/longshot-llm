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
        self.ub_q = ub_q
        self.alpha = alpha
        self.beta = beta
        
        self.n_embed_gate = n_embed_lit * (num_vars + 1)
        self.embedding = torch.nn.Embedding(5, n_embed_lit)
        self.proj1 = torch.nn.Linear(self.n_embed_gate, config.n_embd)
        self.model = GPT2Model(config)
        self.proj2 = torch.nn.Linear(config.n_embd, 1)
        
        self.init_weights()
    
    
    def loss_function(self, y, q):
        """"""
        d = self.ub_q - q
        d_hat = self.ub_q - y
        
        l1 = F.mse_loss(torch.exp(- d_hat), torch.exp(- d))
        l2 = F.mse_loss(y, q)
        loss = self.alpha * l1 + self.beta * l2
        
        return loss

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        attention_mask: torch.LongTensor = None, 
        labels: torch.FloatTensor = None,
        retrun_dict: bool = False
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Optional tensor of shape (batch_size, seq_len)
        
        Returns:
            Tensor of shape (batch_size, seq_len) with predicted Q-values
        """
        # Embed input ids
        x = input_ids
        concat_later = [(x < 0).int()]
        
        for i in range(self.num_vars):
            z = (((x >> i) & 1) | ((x >> (i + 32 - 1)) & 2)) + 2
            concat_later.append(z)
            assert torch.all((z >= 0) & (z < 5)), "Literal encoding out of range"
        
        x = torch.cat([self.embedding(c) for c in concat_later], dim=-1)
        x = self.proj1(x)
        x = self.model(
            inputs_embeds=x, 
            attention_mask=attention_mask, 
            use_cache=False,
            return_dict=False
        )
        y = self.proj2(x).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss = self.loss_function(y, labels)        
            
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
    
    import pprint
    print("Sample Batch:")
    pprint.pprint(sample_batch)
    print("Batch input_ids:\n", batch["input_ids"])
    print("Batch attention_mask:\n", batch["attention_mask"])
    print("Batch labels:\n", batch["labels"])