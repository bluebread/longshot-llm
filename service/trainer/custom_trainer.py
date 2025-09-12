from transformers import Trainer
import torch


class LongshotTrainer(Trainer):
    """
    Custom trainer that logs individual loss components to TensorBoard.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to capture and log individual loss components.
        """
        # Get the standard loss
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
            outputs = None
        
        # Log individual loss components if available
        if hasattr(model, 'loss_components') and model.loss_components:
            # If model is wrapped in DataParallel, access the underlying module
            if hasattr(model, 'module'):
                loss_components = model.module.loss_components
            else:
                loss_components = model.loss_components
            
            # Log to TensorBoard via self.log
            if self.state.global_step > 0:  # Only log after first step
                for key, value in loss_components.items():
                    self.log({f'train/{key}': value})
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, 
                        ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluation loop to log eval loss components.
        """
        # Store accumulated loss components
        eval_loss_components = {
            'l1_mse': [],
            'l2_exp': [],
            'avgQ_loss': [],
            'token_loss': []
        }
        
        # Run standard evaluation
        output = super().evaluation_loop(
            dataloader, 
            description, 
            prediction_loss_only, 
            ignore_keys, 
            metric_key_prefix
        )
        
        # Calculate average eval loss components
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
            
        # Do a quick evaluation pass to collect loss components
        model.eval()
        with torch.no_grad():
            for step, inputs in enumerate(dataloader):
                if step >= 10:  # Sample first 10 batches for component logging
                    break
                    
                inputs = self._prepare_inputs(inputs)
                _ = model(**inputs)
                
                if hasattr(model, 'loss_components') and model.loss_components:
                    for key in eval_loss_components:
                        if key in model.loss_components:
                            eval_loss_components[key].append(model.loss_components[key])
        
        # Log average eval loss components
        for key, values in eval_loss_components.items():
            if values:
                avg_value = sum(values) / len(values)
                self.log({f'{metric_key_prefix}/{key}': avg_value})
        
        return output