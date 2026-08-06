import numpy as np
import torch
from torch.nn import MSELoss, L1Loss


# Motion correction loss
class MC_Loss(torch.nn.Module):
    def __init__(self, dim_rot=3, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.dim_rot = dim_rot

    def forward(self, mc_trafos):
        loss_rot = (mc_trafos[..., :self.dim_rot] ** 2)
        loss_trans = (mc_trafos[..., self.dim_rot:] ** 2)
        loss = loss_rot + 1e-3 * loss_trans
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        return loss

class LinearRegularizer:
    def __init__(self, x_values: list):
        """
        Initializes the linear regularizer by pre-computing the projection matrix
        for the given x_values.

        Args:
            x_values (list or torch.Tensor): A list or 1D Tensor of x-coordinates
                                            for which the y-outputs should lie on a line.
        """
        if not isinstance(x_values, torch.Tensor):
            x_values = torch.tensor(x_values, dtype=torch.float32)
        
        # Ensure x_values is a 1D tensor
        if x_values.dim() != 1:
            raise ValueError("x_values must be a 1D tensor or list.")

        num_points = len(x_values)

        # Construct the design matrix X: [1, x_i]
        # X will have shape (num_points, 2)
        X = torch.stack([torch.ones(num_points, dtype=torch.float32), x_values], dim=1)

        # Calculate X_transpose * X
        XtX = torch.matmul(X.T, X)

        # Calculate (X_transpose * X)^-1
        # Use torch.linalg.inv for robustness if XtX is ill-conditioned
        try:
            XtX_inv = torch.linalg.inv(XtX)
        except RuntimeError as e:
            raise RuntimeError(f"Could not compute inverse of X.T @ X. This might happen if your x_values are all identical or too few points. Error: {e}")

        X_dagger = torch.matmul(XtX_inv, X.T) # Shape: (2, num_points)

        self.projection_matrix = torch.matmul(X, X_dagger) # Shape: (num_points, num_points)
        
        # Pre-compute (P - I)
        self.P_minus_I = self.projection_matrix - torch.eye(num_points, dtype=torch.float32)

        # Move to appropriate device if already set (e.g., if a CUDA device is available)
        self.P_minus_I = self.P_minus_I.to(x_values.device)
        self.projection_matrix = self.projection_matrix.to(x_values.device) # Keep P for potential debugging

    def __call__(self, y: torch.Tensor, sw=None) -> torch.Tensor:
    #def __call__(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the regularization loss for a given batch of y-outputs.

        Args:
            y (torch.Tensor): A tensor of y-outputs from the neural network.
                              Expected shape: (batch_size, num_points) or (num_points,)

        Returns:
            torch.Tensor: The scalar regularization loss (mean squared error
                          over the batch, if y is batched).
        """
        if y.dim() == 1:
            # If y is (num_points,), reshape to (1, num_points) for matrix multiplication
            y = y.unsqueeze(0)
        
        # Ensure y has the correct number of points
        if y.shape[-1] != self.P_minus_I.shape[-1]:
            raise ValueError(f"Number of y-points ({y.shape[-1]}) does not match "
                             f"the number of x_values used for initialization ({self.P_minus_I.shape[-1]}).")

        if self.P_minus_I.device != y.device:
            self.P_minus_I = self.P_minus_I.to(y.device)
        deviation_from_line = torch.matmul(self.P_minus_I, y.T).T # Transpose y for matmul, then transpose back

        loss_per_sample = torch.sum(deviation_from_line**2, dim=-1)

        # Return the mean loss across the batch
        #return torch.mean(loss_per_sample)

        if sw is not None:
            if sw.shape[0] != loss_per_sample.shape[0]:
                raise ValueError(f"Shape mismatch: sw has shape {sw.shape}, expected ({loss_per_sample.shape[0]},)")
            sw = sw.to(loss_per_sample.device)
            return torch.mean((sw) * loss_per_sample)
        else:
            return torch.mean(loss_per_sample)
        
        #return torch.mean(loss_per_sample)


class LossComposition(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.o_r = self.args['outlier_rejection']
        self.loss_fns = {}
        self.loss_fns['values'] = MSELoss(reduction='none') if args['sr_loss_metric'] == 'mse' else L1Loss(reduction='none')
        self.loss_fns['mc'] = MC_Loss(reduction='none')


    def forward(self, v_p, v, v_mask, mc, ss, sw, epoch=0, coeff=0, echotime=0):
        '''
        Args:
            v_p: predicted values, (N, C) // for echo times: (N, 2 (i.e., M0 and T2))
            v: target values, (N, C) // for echo times: (N, C (i.e., number of echo times))
            v_mask: mask to indicate which values are valid, (N, C)
            mc: motion correction parameters, (N, 6) 
            sw: slice weights, (N, 1)
            ss: slice scalings, (N, 1)
        '''
        v_p = v_p * ss if self.o_r else v_p
        loss = self.loss_fns['values'](v_p[v_mask], v[v_mask])
        #if loss nan stop
        regularizer = LinearRegularizer(echotime)

        if self.o_r: #slice weighting is not used right now
            sw = sw.squeeze(-1)
            entropy = -(torch.log(sw + 1e-8)).mean()
            loss = sw * loss + 0.1 * entropy #entropy is used to avoid ssw become 0 and having a pefect loss 0.25 0.5
            loss = loss.mean()
            mc_loss = self.loss_fns['mc'](mc)
            if coeff != 0:
                loss_regul = regularizer(v_p, sw)
                return loss + coeff * loss_regul, loss_regul, mc_loss
            else :
                return loss,None,mc_loss
        
        loss = loss.mean()
        mc_loss = self.loss_fns['mc'](mc)
        #loss= sw * loss + 0.1 * entropy #entropy is used to avoid ssw become 0 and having a pefect loss 0.25 0.5
        if coeff != 0 :
            loss_regul = regularizer(v_p,None)
            return loss + coeff * loss_regul, loss_regul, mc_loss
        else:
            return loss,None,mc_loss
