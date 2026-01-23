import numpy as np
import cv2

# -----------------------------
# Affine Least Squares Estimator
# -----------------------------

def estimate_affine_ls(src, dst):
    """
    src: Nx2 source points
    dst: Nx2 destination points
    """
    N = src.shape[0]
    M = np.hstack([src, np.ones((N,1))])   # Nx3

    bx = dst[:,0]
    by = dst[:,1]

    MtM = M.T @ M
    MtM_inv = np.linalg.inv(MtM)

    ax = MtM_inv @ M.T @ bx
    ay = MtM_inv @ M.T @ by

    H = np.array([
        [ax[0], ax[1], ax[2]],
        [ay[0], ay[1], ay[2]],
        [0,     0,     1]
    ])

    debug = {
        "M": M,
        "MtM": MtM,
        "MtM_inv": MtM_inv,
        "ax": ax,
        "ay": ay
    }

    return H, debug


# -----------------------------
# Homography Estimation
# -----------------------------

def estimate_homography(src, dst):
    H, _ = cv2.findHomography(src.astype(np.float32),
                              dst.astype(np.float32),
                              method=0)
    return H


# -----------------------------
# Apply Transformation
# -----------------------------

def apply_transform(points, H):
    pts_h = np.hstack([points, np.ones((points.shape[0],1))])
    out = (H @ pts_h.T).T
    out = out[:, :2] / out[:, 2:]
    return out