import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from tqdm import trange
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--save_all', action='store_true', help='Save all figures')

args = parser.parse_args()

N = 100
data = ''
version = f'rect_min_targetdiff_nosmooth'
# loss_partial_modal = torch.load(f'schedule/loss_grid_{version.split("_")[0]}{N}.pt')
# loss_partial_modal = torch.load(f'schedule/loss_grid{N}_{version.split("_")[0]}.pt')
loss_partial_modal = torch.load(f'schedule/loss_grid_{version.split("_")[0]}{N}.pt')
# loss_partial_modal[i = t_2d][j = t_3d][key = 'pos' | 'type' | 'bond' | 'loss']


def fit_smooth_surface(Z):
    """
    Fits a smooth surface to a discrete Z grid using B-spline interpolation.

    Args:
        Z (ndarray): 2D grid of loss values.

    Returns:
        spline (RectBivariateSpline): Fitted smooth surface.
    """
    N = Z.shape[0]
    x = np.arange(N)
    y = np.arange(N)
    spline = RectBivariateSpline(x, y, Z, kx=3, ky=3)
    return spline


def flexible_dp_path(Z, N, max_step=3):
    """
    Finds the optimal path on a smooth surface Z with flexible steps and fixed step budget.

    Parameters:
    - Z: 2D array representing the cost surface.
    - N: Total number of steps allowed (budget).
    - max_step: Maximum step size allowed in any direction.

    Returns:
    - optimal_path: List of (x, y) coordinates representing the optimal path.
    - path_cost: Total cost of the optimal path.
    """
    M, M = Z.shape  # Dimensions of the surface
    dp = np.full((M, M, N+1), np.inf)  # DP table
    prev = np.full((M, M, N+1, 2), -1)  # To store previous state

    dp[0, 0, 0] = Z[0, 0]  # Initialize starting point

    for n in trange(N):  # Iterate over steps
        for x in range(M):
            for y in range(M):
                if dp[x, y, n] == np.inf:
                    continue
                for kx in range(0, max_step+1):
                    for ky in range(0, max_step+1):
                        if kx == 0 and ky == 0:
                            continue
                        nx, ny = x + kx, y + ky
                        if 0 <= nx < M and 0 <= ny < M:
                            cost = dp[x, y, n] + Z[nx, ny]
                            if cost < dp[nx, ny, n+1]:
                                dp[nx, ny, n+1] = cost
                                prev[nx, ny, n+1] = [x, y]

    # Find the minimum cost at the endpoint
    optimal_cost = np.inf
    end_x, end_y = -1, -1
    for x in range(M):
        for y in range(M):
            if dp[x, y, N] < optimal_cost:
                optimal_cost = dp[x, y, N]
                end_x, end_y = x, y

    # Backtrack to find the optimal path
    path = []
    cx, cy, cn = end_x, end_y, N
    while cn >= 0:
        path.append((cx, cy))
        cx, cy = prev[cx, cy, cn]
        cn -= 1

    path.reverse()  # Reverse to get the path from start to end
    return np.array(path), optimal_cost

def dynamic_programming_path(Z):
    N = Z.shape[0]
    cost = np.full_like(Z, np.inf)
    prev = np.full((N, N, 2), -1)  # To store previous cell

    cost[0, 0] = Z[0, 0]

    for i in range(N):
        for j in range(N):
            if i > 0:
                new_cost = cost[i-1, j] + Z[i, j]
                if new_cost < cost[i, j]:
                    cost[i, j] = new_cost
                    prev[i, j] = [i-1, j]
            if j > 0:
                new_cost = cost[i, j-1] + Z[i, j]
                if new_cost < cost[i, j]:
                    cost[i, j] = new_cost
                    prev[i, j] = [i, j-1]
            if i > 0 and j > 0:
                new_cost = cost[i-1, j-1] + Z[i, j]
                if new_cost < cost[i, j]:
                    cost[i, j] = new_cost
                    prev[i, j] = [i-1, j-1]

    # Backtrack to reconstruct the path
    path = []
    i, j = N-1, N-1
    while i >= 0 and j >= 0:
        path.append([i, j])
        i, j = prev[i, j]
        if i == -1 or j == -1:
            break

    path = path[::-1]  # Reverse the path
    return np.array(path) / N, cost[N-1, N-1]

def compute_dynamic_steps(Z_smooth):
    """
    Compute dynamic step sizes based on the gradient of the surface.
    """
    grad_x, grad_y = np.gradient(Z_smooth)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    step_size = 1 / (1 + grad_magnitude)
    return step_size

def advanced_flexible_dp(Z_smooth, N):
    """
    Finds the optimal path on a smooth surface Z with dynamic steps and monotonic constraints.

    Parameters:
    - Z_smooth: 2D array representing the smoothed cost surface.
    - N: Total number of steps allowed (budget).

    Returns:
    - optimal_path: List of (x, y) coordinates representing the optimal path.
    - path_cost: Total cost of the optimal path.
    """
    M, M = Z_smooth.shape
    dp = np.full((M, M, N+1), np.inf)  # DP table
    prev = np.full((M, M, N+1, 2), -1)  # To store previous state

    step_size = compute_dynamic_steps(Z_smooth)
    step_size *= M / N  # Scale step size to match the enlarged grid size
    print('Computed dynamic step sizes: max={}, min={}'.format(step_size.max(), step_size.min()))
    dp[0, 0, 0] = Z_smooth[0, 0]  # Initialize starting point

    for n in trange(N):  # Iterate over steps
        for x in range(M):
            for y in range(M):
                if dp[x, y, n] == np.inf:
                    continue
                for kx in range(1, int(step_size[x, y]) + 1):  # Dynamic step in x
                    for ky in range(1, int(step_size[x, y]) + 1):  # Dynamic step in y
                        nx, ny = x + kx, y + ky
                        if nx >= M or ny >= M or kx + ky > step_size[x, y]:
                            continue
                        if nx < x or ny < y:  # Monotonic constraints
                            continue
                        cost = dp[x, y, n] + Z_smooth[nx, ny]
                        if cost < dp[nx, ny, n+1]:
                            dp[nx, ny, n+1] = cost
                            prev[nx, ny, n+1] = [x, y]

    # Find the minimum cost at the endpoint
    optimal_cost = np.inf
    end_x, end_y = -1, -1
    for x in range(M):
        for y in range(M):
            if dp[x, y, N] < optimal_cost:
                optimal_cost = dp[x, y, N]
                end_x, end_y = x, y

    # Backtrack to find the optimal path
    path = []
    cx, cy, cn = end_x, end_y, N
    while cn >= 0:
        path.append((cx, cy))
        cx, cy = prev[cx, cy, cn]
        cn -= 1

    path.reverse()  # Reverse to get the path from start to end
    return np.array(path), optimal_cost

def advanced_flexible_dp_with_checks(Z_smooth, N):
    M, M = Z_smooth.shape
    dp = np.full((M, M, N+1), np.inf)  # DP table
    prev = np.full((M, M, N+1, 2), -1)  # To store previous state

    step_size = compute_dynamic_steps(Z_smooth)
    step_size *= np.sqrt(2)  # Increase step size to allow diagonal steps
    # step_size *= 2  # Increase step size to allow more flexibility
    print('Computed dynamic step sizes: max={}, min={}'.format(step_size.max(), step_size.min()))
    dp[0, 0, 0] = Z_smooth[0, 0]  # Initialize starting point

    if 'dock' in version:
        # set the first row to be directly reachable from the first cell
        for i in range(1, M):
            dp[i, 0, 0] = Z_smooth[i, 0]

    cnt = 0
    for n in trange(N):  # Iterate over steps
        if n % 10 == 0 or n == N-1:
            print('Current min cost: {} @[{}], destination min cost: {} @[{}]'.format(dp[:, :, n].min(), n, dp[M-1, M-1, :].min(), np.argmin(dp[M-1, M-1])))
        for x in range(M):
            for y in range(M):
                cnt += 1
                if dp[x, y, n] == np.inf:
                    continue
                flag = False
                for kx in range(0, min(M-x, int(step_size[x, y]) + 2)):
                    for ky in range(0, min(M-y, int(step_size[x, y]) + 2)):
                        nx, ny = x + kx, y + ky
                        if nx >= M or ny >= M:
                            continue
                        if nx < x or ny < y:  # Monotonicity
                            continue
                        if kx + ky > step_size[x, y] + 1:
                            continue
                        # if nx == 0 or ny == 0:
                        #     continue
                        if kx + ky == 0:
                            continue
                        # if kx > 0 or ky > 0:
                        #     cost = dp[x, y, n] + Z_smooth[nx, ny]
                        # else:
                        #     cost = dp[x, y, n]
                        cost = dp[x, y, n] + Z_smooth[nx, ny]
                        
                        # if nx == M-1 and ny == M-1:
                        #     print(f"Updating dp[{nx}, {ny}, {n+1}] from dp[{x}, {y}, {n}] with cost {cost}")
                        
                        # if n == N-1 and (nx, ny) == (M-1, M-1):
                        #     print(f"Reached endpoint ({nx}, {ny}) with cost {dp[nx, ny, n+1]} updated from {cost}")

                        # print('cost: {}'.format(cost))
                        if cost < dp[nx, ny, n+1]:
                            flag = True
                            dp[nx, ny, n+1] = cost
                            prev[nx, ny, n+1] = [x, y]
                if not flag:
                    pass
                    # print('No valid step found for ({}, {}) at step {} (step size = {} -> {})'.format(x, y, n, step_size[x, y], int(step_size[x, y]) + 2))

    # Check if endpoint is reachable within the budget
    final_N = np.argmin(dp[M-1, M-1])
    if dp[M-1, M-1, final_N] == np.inf:
        print(dp[M-3:M, M-3:M, N-3:N+1])
        breakpoint()
        raise ValueError("No valid path found to (M-1, M-1).")
    
    # Pick the final_N as the closest step to M and is not infinity
    if 'closest' in version:
        dist_to_M = np.abs(M - final_N)
        for i in range(N, 0, -1):
            if dp[M-1, M-1, i] != np.inf:
                dist = np.abs(M - i)
                if dist < dist_to_M:
                    final_N = i
                    dist_to_M = dist


    # Backtrack to find the optimal path
    path = []
    cx, cy, cn = M-1, M-1, final_N
    while cn >= 0:
        path.append((cx, cy))
        if cx == 0 and cy == 0:
            break
        if 'dock' in version and cy == 0:
            break
        cx, cy = prev[cx, cy, cn]
        cn -= 1

    # Check if backtracking reached the start
    if (cx, cy) != (0, 0):
        if not ('dock' in version and cy == 0):
            breakpoint()
            raise ValueError("Backtracking failed to reach (0, 0).")

    path.reverse()  # Reverse to get the path from start to end
    print('final_N: {}, optimal cost: {} (M = {})'.format(final_N, dp[M-1, M-1, final_N], M))
    return np.array(path), dp[M-1, M-1, final_N]


def plot_surface(ax, Z):
    N = Z.shape[0]
    t = list(range(N))
    X, Y = np.meshgrid(t, t)
    Z = Z.reshape(N, N)
    

def plot_optimized_path(spline, Z, optimized_path):
    """
    Plots the smooth surface and the optimized path.

    Args:
        spline (RectBivariateSpline): Fitted smooth surface.
        Z (ndarray): Original discrete grid.
        optimized_path (ndarray): Optimized path coordinates.
    """
    N = Z.shape[0]
    t = list(range(N))
    t_id = np.array(t)
    t_id = t_id / t_id.max()
    X, Y = np.meshgrid(t_id, t_id, indexing='ij')
    Z_smooth = spline(t, t)
    
    # Plot surface
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 10, 'legend.fontsize': 20})
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if 'rect' in version:
        mask = None
    else:
        mask = X >= Y
        mask = None
    # mask = None
    if mask is None:
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    else:
        # Flatten and mask
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        valid_indices = mask.flatten()

        # Filter valid points
        X_valid = X_flat[valid_indices]
        Y_valid = Y_flat[valid_indices]
        Z_valid = Z_flat[valid_indices]

        # Plot with trisurf for the valid region
        ax.plot_trisurf(X_valid, Y_valid, Z_valid, cmap='viridis', alpha=0.8)
    
    # Plot path
    path_x, path_y = optimized_path[:, 0], optimized_path[:, 1]
    # clamp path_x, path_y to be within the grid
    # path_x = np.clip(path_x, 0, N-1)
    # path_y = np.clip(path_y, 0, N-1)
    path_z = spline.ev(path_x, path_y)
    # change the path_z to be the same as the original Z
    path_z = Z[path_x.astype(int), path_y.astype(int)]

    path_x = np.array(path_x).astype(float)
    path_x /= path_x.max()
    path_y = np.array(path_y).astype(float)
    path_y /= path_y.max()

    ax.plot(path_x, path_y, path_z, color='tab:red', linewidth=2, label='Optimal Path')
    
    # sanity check
    ax.plot(path_x, path_y, np.zeros_like(path_z), color='black', linewidth=2, linestyle='--', label='Projected Path')
    # ax.plot(path_x, np.zeros_like(path_y), np.zeros_like(path_z), color='blue', linewidth=3, label='y=0, z=0')
    # ax.plot(path_x, np.zeros_like(path_y), Z[path_x.astype(int), np.zeros_like(path_y)], color='green', linewidth=3, label='y=0')
    # ax.plot(np.zeros_like(path_x), path_y, Z[np.zeros_like(path_x), path_y.astype(int)], linewidth=3, label='x=0')
    # ax.plot(np.zeros_like(path_x), path_y, np.zeros_like(path_z), linewidth=3, color='pink', label='x=0, z=0')
    
    ax.set_xlabel(r'$t_d$ (discrete)', fontsize=20)
    ax.set_ylabel(r'$t_c$ (continuous)', fontsize=20)
    ax.set_zlabel('Loss', fontsize=20)
    ax.legend()

Z = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        # Z[i, j] = loss_partial_modal[i][j]['loss']
        Z[j, i] = loss_partial_modal[i][j]['pos'] + loss_partial_modal[i][j]['type'] * 100  # loss_v_weight
        # for key in ['pos', 'type']:
        #     Z[i, j] += loss_partial_modal[i][j][key]

# Z = np.array([[loss_partial_modal[i][j]['loss'] for j in range(N)] for i in range(N)])

# fill the zero values with np.nan
Z[Z == 0] = 20

N = Z.shape[0]
Z = Z.reshape(N, N)
# clamp loss max
Z = np.clip(Z, 0, 20)
spline = fit_smooth_surface(Z)
print('Fitted smooth surface')

save_all = args.save_all

if save_all:
    # visualize the smooth surface
    t = np.linspace(0, N-1, 1000)
    X, Y = np.meshgrid(t, t)
    Z_smooth = spline(t, t)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z_smooth, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(f'surface_smooth_{version}.png')

    # visualize the original surface
    t = list(range(N))
    X, Y = np.meshgrid(t, t)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    plt.savefig(f'surface_original_{version}.png')
    
    df = pd.DataFrame(Z)
    df.to_csv('loss_grid.csv', index=True, header=True)


# optimized_path, cost, cost_init = optimize_path(spline, N, num_points=1000, path_constraint=False)
# optimized_path, cost = dynamic_programming_path(Z_smooth)
# optimized_path, cost = flexible_dp_path(Z_smooth, N=100, max_step=5)
# optimized_path, cost = advanced_flexible_dp(Z_smooth, N=1000)

input_z = Z
# input_z = Z_smooth

import os
if False:
# if os.path.exists(f'optimized_path_{version}.pt'):
    optimized_path = torch.load(f'optimized_path_{version}.pt')
    cost = 0
else:
    optimized_path, cost = advanced_flexible_dp_with_checks(input_z, N=int(1.2 * len(input_z)))
    # optimized_path, cost = advanced_flexible_dp_with_checks(input_z, N=100)
    print('Optimized path found')
    torch.save(optimized_path, f'optimized_path_{version}.pt')

plot_optimized_path(spline, input_z, optimized_path)
# plt.title(version)
plt.savefig(f'loss_geodesic_{version}.svg', dpi=300, bbox_inches='tight')


# visualize the 2d path
plt.figure()
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 14})
optimized_path = optimized_path.astype(float)
optimized_path /= optimized_path.max()
t = np.linspace(0, 1, len(optimized_path))
plt.plot(t, optimized_path[:, 0], label=r'$t_d$ (discrete)')
plt.plot(t, optimized_path[:, 1], label=r'$t_c$ (continuous)')
plt.xlabel(r'$t$')
# plt.ylabel('t')
# plt.title(version)
plt.legend()
plt.savefig(f'path_geodesic_{version}.pdf', dpi=300, bbox_inches='tight')

# initial cost along the diagonal
if save_all:
    cost_init = np.sum([input_z[i, i] for i in range(input_z.shape[0])])
    print(cost, cost_init, version)
