## generate pseudo bulk
from numba import jit, prange
import numpy as np
import pandas as pd

@jit(nopython=True, parallel=True)
def generate_single_bulk(data, indices, cell_counts, num_genes, n_bulk_samples):
    bulk_samples = np.zeros((n_bulk_samples, num_genes))
    for i in prange(n_bulk_samples):
        bulk_sample = np.zeros(num_genes)
        for cell_type in prange(len(indices)):
            idx_to_sample = np.random.choice(indices[cell_type], cell_counts[cell_type], replace=True)
            bulk_sample += data[idx_to_sample].sum(axis=0)
        bulk_samples[i] = bulk_sample
    return bulk_samples

def generate_bulk_matrix(single_cell_matrix, cell_type_vector, n_bulk_samples, cells_per_bulk):
    data = single_cell_matrix.values
    columns = single_cell_matrix.columns
    cell_types = single_cell_matrix.index.unique()
    type_indices = [np.where(single_cell_matrix.index == ct)[0] for ct in cell_types]

    # Precompute cell counts for each cell type
    cell_counts = np.array([round(proportion * cells_per_bulk) for proportion in cell_type_vector.values])

    # Sample and sum the data
    bulk_samples = generate_single_bulk(data, type_indices, cell_counts, single_cell_matrix.shape[1], n_bulk_samples)

    # Normalize all bulk samples at once
    bulk_samples /= bulk_samples.sum(axis=1, keepdims=True)

    return pd.DataFrame(bulk_samples, index=  ['sample{}'.format(i + 1) for i in range(n_bulk_samples)], columns=columns)


# generate pseudo_bulk ct proportion dataframe
def generate_k_matrix(p, n_bulk_samples):

    p_array = p.values
    K = np.tile(p_array, (n_bulk_samples, 1)).T
    return pd.DataFrame(K,index=p.index,columns = ['sample{}'.format(i + 1) for i in range(n_bulk_samples)])


# calculate loss wethin pytorch
import torch
import pandas as pd
import numpy as np

def initialize_tensors(Y, K, X, w):
    """
    Convert input from Pandas DataFrame and NumPy array to PyTorch tensors and ensure data type is float.
    """
    Y = torch.tensor(Y.values, dtype=torch.float64) if isinstance(Y, pd.DataFrame) else torch.tensor(Y, dtype=torch.float64)
    K = torch.tensor(K.values, dtype=torch.float64) if isinstance(K, pd.DataFrame) else torch.tensor(K, dtype=torch.float64)
    X = torch.tensor(X.values, dtype=torch.float64) if isinstance(X, pd.DataFrame) else torch.tensor(X, dtype=torch.float64)
    w = torch.tensor(w, dtype=torch.float64, requires_grad=True)  # Set requires_grad=True for w
    return Y, K, X , w

def compute_loss(Y, K, X, w, lambda1,  delta, sigma):  #gama,
    KX = X.matmul(K)
    mse_grad = compute_gradient(KX, Y, X, w)
    loss1 = (mse_grad ** 2).sum()

    gradient_change = calculate_gradient_change(Y, K, X, w, delta, sigma)   #gama,
    # loss2 = torch.sum(torch.stack(gradient_change))    #### min改为sum
    loss2 = torch.min(torch.stack(gradient_change))    #### min

    total_loss = lambda1 * loss1 - (1 - lambda1) * loss2
    # print(f'loss1:{loss1}   loss2:{loss2}')
    return total_loss

def compute_gradient(KX, Y, X, w):
    differences = KX - Y.t()
    grad_mse = 2 * X.t().matmul(differences * w[:, None])
    return grad_mse

def perturbation_gradient_change(Y, k_plus, k_minus, X, w):
    KX_plus = X.matmul(k_plus)
    KX_minus = X.matmul(k_minus)
    return (compute_gradient(KX_plus, Y, X, w) - compute_gradient(KX_minus, Y, X, w)).sum()

def absolute_perturbation_gradient(Y, K, X, w, delta, i):
    k_plus_absolute = K.clone()
    k_plus_absolute[i] += delta
    k_minus_absolute = K.clone()
    k_minus_absolute[i] -= delta
    # print(f'absolut loss: {perturbation_gradient_change(Y, k_plus_absolute, k_minus_absolute, X, w)}')
    return perturbation_gradient_change(Y, k_plus_absolute, k_minus_absolute, X, w)

def relative_perturbation_gradient(Y, K, X, w, sigma, i):
    k_plus_relative = K.clone()
    k_plus_relative[i] += K[i, 0] * sigma
    k_minus_relative = K.clone()
    k_minus_relative[i] -= K[i, 0] * sigma
    # print(f'relative loss: {perturbation_gradient_change(Y, k_plus_relative, k_minus_relative, X, w)}')
    return perturbation_gradient_change(Y, k_plus_relative, k_minus_relative, X, w)

def calculate_gradient_change(Y, K, X, w,  delta, sigma):  #gama,
    grad_change_ct = []
    for i in range(K.shape[0]):
        absolute_grad = absolute_perturbation_gradient(Y, K, X, w, delta, i)
        relative_grad = relative_perturbation_gradient(Y, K, X, w, sigma, i)
        # grad_cti = gama * absolute_grad + (1 - gama) * relative_grad
        grad_cti = torch.min(absolute_grad, relative_grad)
        grad_change_ct.append(grad_cti)
    return grad_change_ct



## optimize weight
def weight_optimization( w_tensor, Y_tensor, K_tensor, X_tensor,lr=0.01, lambda1=0.8, delta=1e-5, sigma=1e-2, num_epochs = 1000, tolerance = 1e-6): #gama=0.5,
    
    optimizer = torch.optim.Adam([w_tensor], lr=lr)
    # Training loop
    unchanged_count = 0
    # prev_w = w_tensor.clone()

    prev_loss = 0 
    # loss_unchanged_count = 0
    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()  # Clear gradients
    #     loss = compute_loss(Y_tensor, K_tensor, X_tensor, w_tensor, lambda1=0.5, gama=0.5, delta=1e-5, sigma=1e-2)
    #     loss.backward()  # Compute gradients
    #     optimizer.step()  # Update parameters
    
    #     # Ensure non-negative values and normalize to sum to 1
    #     with torch.no_grad():
    #         w_tensor.clamp_(min=0)  # Clamp w_tensor to ensure all values are non-negative
    #         w_tensor /= w_tensor.sum()  # Normalize w_tensor so that its values sum to 1

    #     # Check if loss has changed significantly
    #     if prev_loss is not None and abs(loss.item() - prev_loss) < tolerance:
    #         loss_unchanged_count += 1
    #     else:
    #         loss_unchanged_count = 0

    #     prev_loss = loss.item()

    #     if loss_unchanged_count >= 5:
    #         print(f'Stopping early at epoch {epoch + 1} due to loss not changing significantly.')
    #         break

    #     if (epoch + 1) % 100 == 0:
    #         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    #         # print(f'Updated w_tensor: {w_tensor.detach().numpy()}')

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Clear gradients
        loss = compute_loss(Y_tensor, K_tensor, X_tensor, w_tensor, lambda1, delta, sigma)   #gama, 
        # print(loss)
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
    
        # Ensure non-negative values and normalize to sum to 1
        with torch.no_grad():
            w_tensor.clamp_(min=0)  # Clamp w_tensor to ensure all values are non-negative
            w_tensor /= w_tensor.sum()  # Normalize w_tensor so that its values sum to 1

        # # Check if w_tensor has changed
        # if torch.allclose(w_tensor, prev_w, atol=tolerance):
        #     unchanged_count += 1
        # else:
        #     unchanged_count = 0

        # Check if loss has changed
        if abs(loss.item()-prev_loss<tolerance):
            unchanged_count += 1
        else:
            unchanged_count = 0

        prev_loss = loss.item()
        # prev_w = w_tensor.clone()

        if unchanged_count >= 5:
            # print(f'Stopping early at epoch {epoch + 1} due to w_tensor not changing.')
            break

        # if (epoch + 1) % 100 == 0:
        #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            # print(f'Updated w_tensor: {w_tensor.detach().numpy()}')

    # Print final optimized w_tensor
    # print("Optimized w_tensor:", w_tensor.detach().numpy())
    return w_tensor

