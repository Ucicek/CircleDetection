import torch

def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Mean Absolute Error (MAE) loss between predicted and target values for circle parameters.

    Args:
        pred (torch.Tensor): Predictions tensor with shape [batch_size, 3], where each row contains 
                             [x_pred, y_pred, r_pred].
        target (torch.Tensor): Target tensor with shape [batch_size, 3], where each row contains 
                               [x_target, y_target, r_target].

    Returns:
        torch.Tensor: The total mean absolute error as a single scalar tensor, combining the MAE of 
                      x, y, and radius (r) predictions.
    """
    x_pred, y_pred, r_pred = pred[:, 0], pred[:, 1], pred[:, 2]
    x_target, y_target, r_target = target[:, 0], target[:, 1], target[:, 2]
    x_error = torch.mean(torch.abs(x_pred - x_target))
    y_error = torch.mean(torch.abs(y_pred - y_target))
    r_error = torch.mean(torch.abs(r_pred - r_target))
    return x_error + y_error + r_error

