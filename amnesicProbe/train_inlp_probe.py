import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def build_inlp_projection(X, Y, num_iters=50, random_acc_threshold=0.02):
    """
    1) 先做 label => int
    2) random_baseline_acc = 1/num_classes
    3) 多次迭代:
       - 训练线性分类器 => 如果acc在random基线+threshold之内 => early stop
       - 否则移除对应1维
    """
    label2id = {label: i for i, label in enumerate(set(Y))}
    Y_int = np.array([label2id[lbl] for lbl in Y], dtype=np.int64)
    hidden_dim = X.shape[1]

    num_classes = len(label2id)
    random_baseline_acc = 1.0 / num_classes
    print(f"Random baseline accuracy: {random_baseline_acc:.4f}")

    P = np.eye(hidden_dim, dtype=np.float32)

    for it in range(num_iters):
        XP = X @ P  # shape (N, hidden_dim_reduced)
        clf = SGDClassifier(loss="hinge", max_iter=5000, tol=1e-3)
        clf.fit(XP, Y_int)

        # predict
        Y_pred = clf.predict(XP)
        acc = accuracy_score(Y_int, Y_pred)
        print(f"Iteration {it+1}, accuracy={acc:.4f}")

        if acc - random_baseline_acc < random_acc_threshold:
            print(f"Early stop at iteration {it+1}. Acc near random.")
            break

        # get weight
        w = clf.coef_[0]
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-9:
            print("Weight norm ~ 0, stopping.")
            break
        w_normed = w / w_norm

        # remove dimension
        w_2d = np.outer(w_normed, w_normed)
        P = P @ (np.eye(hidden_dim, dtype=np.float32) - w_2d)
        print("Removed 1 dimension from subspace.")

    return P

def main():
    input_npz = "sampled_hidden_representations.npz"
    output_projection = "pos_amnesic_projection.npy"

    data = np.load(input_npz, allow_pickle=True)
    X = data["hidden"]  # shape (N, 4096)
    Y = data["pos"]     # shape (N,)

    print(f"X={X.shape}, Y={Y.shape}")
    X = X.astype(np.float32, copy=False)

    # PCA
    pca_dim = 256
    print(f"Doing PCA -> {pca_dim} dims.")
    pca = PCA(n_components=pca_dim, svd_solver='randomized')
    X_pca = pca.fit_transform(X)

    # INLP
    P_inlp = build_inlp_projection(X_pca, Y, num_iters=50, random_acc_threshold=0.02)
    # P_inlp shape => (pca_dim, pca_dim)

    # save
    np.save("pca_components.npy", pca.components_)
    np.save("pca_mean.npy", pca.mean_)
    np.save(output_projection, P_inlp)
    print("Done.")

if __name__=="__main__":
    main()
