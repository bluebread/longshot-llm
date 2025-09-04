from typing import List, Dict, Any
import torch
from torch.nn.utils.rnn import pad_sequence


class TrajectoryCollator:
    """
    Data collator for TrajectoryDataset that forms batches for Hugging Face training.
    
    This collator handles variable-length sequences by padding them to the same length
    and creating appropriate attention masks.
    """
    
    def __init__(
        self,
        num_vars: int,
        padding_value: int = 0,
        label_padding_value: float = -100.0,
        return_tensors: str = "pt",
        permute_input: bool = False,
    ):
        """
        Initialize the TrajectoryCollector.
        
        Args:
            padding_value: Value to use for padding input_ids (default: 0)
            label_padding_value: Value to use for padding labels (default: -100.0 for loss ignoring)
            return_tensors: Type of tensors to return ("pt" for PyTorch)
        """
        self.num_vars = num_vars
        self.padding_value = padding_value
        self.label_padding_value = label_padding_value
        self.return_tensors = return_tensors
        self.permute_input = permute_input
    
    
    def _random_permutation(self) -> torch.Tensor:
        n = self.num_vars
        
        perm = torch.randperm(n, dtype=torch.int64).repeat(2)
        sgn = torch.randint(0, 2, (n,), dtype=torch.int64)
        perm[:n] += sgn * n
        perm[n:] += (1 - sgn) * n
            
        return perm


    def _convert_to_binary(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embed input ids
        x = torch.abs(input_ids)
        concat_later = []
        
        for i in range(self.num_vars):
            pos_z = ((x >> i) & 0x1).unsqueeze(-1)
            concat_later.append(pos_z)
        
        for i in range(self.num_vars):
            neg_z = ((x >> (i + 32)) & 0x1).unsqueeze(-1)
            concat_later.append(neg_z)
        
        return torch.cat(concat_later, dim=-1)
    
    
    def _permute_tensor(self, input_bin: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        return input_bin.index_select(-1, perm)


    def _convert_to_ids(self, input_bin: torch.Tensor) -> torch.Tensor:
        x = input_bin
        pos_w = [(1 << i) for i in range(self.num_vars)]
        neg_w = [(1 << (i + 32)) for i in range(self.num_vars)]
        w = torch.tensor(pos_w + neg_w, dtype=torch.int64, device=input_bin.device)
        y = x @ w
        return y.squeeze(-1)


    def _permute(self, input_ids: torch.Tensor) -> torch.Tensor:
        sign = 1 - 2 * (input_ids < 0).int()
        input_bin = self._convert_to_binary(input_ids)
        perm = self._random_permutation()
        permuted_bin = self._permute_tensor(input_bin, perm)
        return self._convert_to_ids(permuted_bin) * sign


    def __call__(self, features: List[Dict[str, Any]], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples from TrajectoryDataset into a batch.
        
        Args:
            features: List of samples from TrajectoryDataset, each containing:
                - input_ids: List of token IDs (variable length)
                - attention_mask: List of attention mask values
                - labels: List of Q-values or target values
        
        Returns:
            Dictionary containing batched tensors:
                - input_ids: Padded input tensor of shape (batch_size, max_seq_len)
                - attention_mask: Attention mask tensor of shape (batch_size, max_seq_len)
                - labels: Padded labels tensor of shape (batch_size, max_seq_len)
        """
        # Extract input_ids, attention_masks, and labels from features
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.int64) for f in features]
        attention_masks = [torch.tensor(f["attention_mask"], dtype=torch.int64) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.float32) for f in features]
        
        # Pad sequences to the same length
        # pad_sequence pads to the length of the longest sequence in the batch
        input_ids_padded = pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.padding_value
        )
        
        attention_mask_padded = pad_sequence(
            attention_masks, 
            batch_first=True, 
            padding_value=0  # Attention mask should be 0 for padded positions
        )
        
        labels_padded = pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=self.label_padding_value
        )
        
        if self.permute_input:
            input_ids_padded = self._permute(input_ids_padded)
        
        batch = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded
        }
        
        return batch

if __name__ == "__main__":
    # Example usage
    from dataset import TrajectoryDataset
    
    dataset = TrajectoryDataset(num_vars=3, width=2)
    collector = TrajectoryCollator()
    sample_batch = [dataset[i] for i in range(4)]  # Get 4 samples
    batch = collector(sample_batch)
    
    import pprint
    print("Sample Batch:")
    pprint.pprint(sample_batch)
    print("Batch input_ids:\n", batch["input_ids"])
    print("Batch attention_mask:\n", batch["attention_mask"])
    print("Batch labels:\n", batch["labels"])