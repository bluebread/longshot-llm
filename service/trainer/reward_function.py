import math
import warnings
from typing import Union, List
import torch
from longshot.formula import DNF, NormalFormFormula
from longshot.literals import Literals

class LongshotRewardFunction:
    """
    Reward function for evaluating Longshot formula generation quality.

    This reward function evaluates the quality of generated boolean formula
    sequences based on their average Q values and validity. It's designed
    for use with reinforcement learning frameworks like TRL for fine-tuning
    language models on formula generation tasks.

    Attributes:
        num_vars: Number of variables in the boolean formulas
        width: Maximum width (number of literals) per term
        penalty: Penalty score for invalid tokens
        ub_q: Upper bound for Q values (used in exponential scaling)
        alpha: Weight for exponential reward component
        beta: Weight for linear reward component
    """

    def __init__(
        self,
        num_vars: int,
        width: int,
        penalty: float,  # Fixed typo
        ub_q: float,
        alpha: float,
        beta: float,
    ):
        self.num_vars = num_vars
        self.width = width
        self.penalty = penalty  # Fixed typo
        self.ub_q = ub_q
        self.alpha = alpha
        self.beta = beta
        
        self.n_choose_w = math.comb(num_vars, width)
        self.pow2_w = 2 ** width


    def _index_to_subset(self, m: int) -> List[int]:
        """
        Convert an index m to a k-subset of n elements.

        This implements combinatorial indexing to map an integer to a unique
        combination of k elements from n total elements.

        Args:
            m: Index number (0-based) representing the subset

        Returns:
            List of indices representing the subset

        Example:
            For n=4, k=2: m=0 -> [1,2], m=1 -> [1,3], m=2 -> [1,4], etc.
        """
        n = self.num_vars
        k = self.width
        
        subset = []
        c = n
        
        for i in range(k):
            # Find elements from largest to smallest
            c = c - 1
            while c >= k - i - 1:
                count = math.comb(c, k - i - 1)  # C(c, k-i-1)
                if m >= count:
                    m -= count
                    c -= 1
                else:
                    break
            subset.append(n - c)
            
        return subset


    def _decode_token_id(self, token: int) -> tuple[int, int, int]:
        """
        Decode a token ID into its components.

        Args:
            token: Token ID to decode

        Returns:
            Tuple of (token_type, positive_literals, negative_literals)
            - token_type: 0 for ADD, 1 for DELETE
            - positive_literals: Bitmask of positive literals
            - negative_literals: Bitmask of negative literals
        """
        tt = token // (self.n_choose_w * self.pow2_w)
        idx = (token % (self.n_choose_w * self.pow2_w)) // (self.pow2_w)
        sgn = token % self.pow2_w
        vars = self._index_to_subset(idx) # variables
        pos = 0
        neg = 0
        
        for i, v in enumerate(vars):
            if (sgn & (1 << i)) == 0:  # positive literal
                pos |= (1 << v)
            else:
                neg |= (1 << v)
        
        return tt, pos, neg


    def _is_invalid_token(self, f: NormalFormFormula, tt: int, lit: Literals) -> bool:
        """
        Check if a token operation would be invalid.

        Args:
            f: Current formula state
            tt: Token type (0=ADD, 1=DELETE)
            lit: Literals to add or delete

        Returns:
            True if the operation is invalid, False otherwise
        """
        if lit.is_constant or lit.width > self.width:
            return True
        elif lit in f and tt == 0:
            return True
        elif lit not in f and tt == 1:
            return True
        return False


    def _init_formula_from_prompt(self, prompt: list[int]) -> tuple[NormalFormFormula, float]:
        """
        Initialize a formula from a prompt sequence.

        Args:
            prompt: List of literal integers representing the initial formula

        Returns:
            Tuple of (formula, max_avgQ) where:
            - formula: The constructed NormalFormFormula
            - max_avgQ: Maximum average Q value achieved during construction
        """
        formula = DNF(self.num_vars)
        prompt_maxQ = 0
        
        for i, litint in enumerate(prompt):
            tt = int(litint > 0)
            x = abs(litint)
            pos = x & 0xFFFFFFFF
            neg = (x >> 32) & 0xFFFFFFFF
            lit = Literals(pos, neg)
            
            if self._is_invalid_token(formula, tt, lit):
                ts = 'ADD' if tt == 0 else 'DEL'
                warnings.warn(f"Prompt contains invalid token: {ts} {x} (i={i})")
                continue
            
            formula.toggle(lit)
            prompt_maxQ = max(prompt_maxQ, formula.avgQ())
            
        return formula, prompt_maxQ

    
    def _calculate_score(self, reply_maxQ: float, prompt_maxQ: float, total_penalty: float) -> float:
        """
        Calculate the final reward score.

        The reward combines exponential and linear components based on Q-value
        improvements, minus any penalties for invalid tokens.

        Args:
            reply_maxQ: Maximum Q value achieved in the generated sequence
            prompt_maxQ: Maximum Q value from the prompt
            total_penalty: Accumulated penalty for invalid tokens

        Returns:
            Final reward score
        """
        E = math.exp(prompt_maxQ - self.ub_q)
        E_hat = math.exp(reply_maxQ - self.ub_q)
        Q = prompt_maxQ
        Q_hat = reply_maxQ
        r1 = self.alpha * (E_hat - E)
        r2 = self.beta * (Q_hat - Q)
        
        return r1 + r2 - total_penalty
    

    def score(self, prompt: list[int], sequence: list[int]) -> float:
        """
        Compute reward score for a generated sequence given a prompt.

        Args:
            prompt: Initial formula as list of literal integers
            sequence: Generated token sequence to evaluate

        Returns:
            Reward score (higher is better)
        """
        tokens = [self._decode_token_id(ti) for ti in sequence]
        formula, prompt_maxQ = self._init_formula_from_prompt(prompt)
        total_penalty = 0
        reply_maxQ = 0
        
        for tt, pos, neg in tokens:
            lit = Literals(pos, neg)
            
            if self._is_invalid_token(formula, tt, lit):
                total_penalty -= self.penalty
            else:
                formula.toggle(lit)
                reply_maxQ = max(reply_maxQ, formula.avgQ())

        return self._calculate_score(reply_maxQ, prompt_maxQ, total_penalty)

    def compute_batch_rewards(
        self,
        prompts: Union[List[List[int]], torch.Tensor],
        sequences: Union[List[List[int]], torch.Tensor],
        return_tensor: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, List[float]]:
        """
        Compute rewards for a batch of prompts and sequences (TRL integration).

        Args:
            prompts: Batch of prompt token sequences (batch_size, prompt_length)
            sequences: Batch of generated sequences (batch_size, seq_length)
            return_tensor: If True, return as tensor; if False, return as list

        Returns:
            rewards: Tensor of reward scores (batch_size,) or list of floats
        """
        # Convert tensors to lists for processing
        if isinstance(prompts, torch.Tensor):
            prompts = prompts.cpu().tolist()
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.cpu().tolist()

        # Handle single sample case
        if not isinstance(prompts[0], list):
            prompts = [prompts]
            sequences = [sequences]

        # Compute rewards for batch
        rewards = []
        for prompt, sequence in zip(prompts, sequences):
            # Filter out padding tokens (0 or negative values indicate padding)
            prompt = [t for t in prompt if t > 0]
            sequence = [t for t in sequence if t > 0]

            try:
                reward = self.score(prompt, sequence)
            except Exception as e:
                warnings.warn(f"Error computing reward: {e}. Using penalty score.")
                reward = -self.penalty * len(sequence)  # Fallback penalty

            rewards.append(reward)

        if return_tensor:
            return torch.tensor(rewards, dtype=torch.float32)
        return rewards

    def __call__(
        self,
        prompts: Union[List[List[int]], torch.Tensor],
        sequences: Union[List[List[int]], torch.Tensor],
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Make the reward function callable for TRL PPOTrainer compatibility.
        Returns list of scalar tensors for each sample.
        """
        rewards = self.compute_batch_rewards(prompts, sequences, return_tensor=True, **kwargs)
        # Return as list of scalar tensors for TRL
        return [r.unsqueeze(0) for r in rewards]



# Example usage with TRL:
# from trl import PPOTrainer, PPOConfig
# from transformers import AutoTokenizer
# from service.trainer.model import GPT2ForLongshot
#
# # Initialize model and tokenizer
# model = GPT2ForLongshot.from_pretrained("path/to/model")
# tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")
#
# # Initialize reward function
# reward_fn = LongshotRewardFunction(
#     num_vars=4,
#     width=3,
#     penalty=0.1,
#     ub_q=1.0,
#     alpha=0.5,
#     beta=0.5,
# )
#
# # Configure PPO
# ppo_config = PPOConfig(
#     model_name="gpt2-longshot",
#     batch_size=16,
#     learning_rate=1e-5,
# )
#
# # Create PPO trainer
# ppo_trainer = PPOTrainer(
#     config=ppo_config,
#     model=model,
#     tokenizer=tokenizer,
#     reward_model=reward_fn,  # Pass reward function directly
# )
