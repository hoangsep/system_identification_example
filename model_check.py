import pickle, torch, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import train_model as tm

# Load data and scalers
X_raw, Y_raw = tm.load_all_and_process(tm.DATA_DIR)
with open(tm.SCALER_SAVE_PATH,'rb') as f: s = pickle.load(f)
sx, sy = s['x'], s['y']
X, Y = sx.transform(X_raw), sy.transform(Y_raw)
_, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Load model
m = tm.DynamicsModel(5,4)
m.load_state_dict(torch.load(tm.MODEL_SAVE_PATH, map_location='cpu'))
m.eval()

X_t = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    pred_scaled = m(X_t).numpy()
pred = sy.inverse_transform(pred_scaled)
actual = sy.inverse_transform(Y_test)
err = pred - actual
rmse = np.sqrt(np.mean(err**2, axis=0))
bias = np.mean(err, axis=0)
print("RMSE dx dy dyaw dv:", rmse)
print("Bias  dx dy dyaw dv:", bias)
