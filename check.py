import torch, numpy as np, onnxruntime as ort, onnx
from onnx import numpy_helper
from my_kws.cached_datasets import CachedKeywordSpottingDataset
from my_kws.model import DSCNN

# --- 疑点1: fp32 权重 dtype 和真实大小 ---
m_onnx = onnx.load('models/dscnn_fp32.onnx')
arrs = [numpy_helper.to_array(i) for i in m_onnx.graph.initializer]
print('n_params:', sum(a.size for a in arrs))
print('dtypes:', {a.dtype.name for a in arrs})
print('weight bytes:', sum(a.nbytes for a in arrs))

# --- 疑点2: 分批算 PT vs ONNX 全量分数, 找翻面样本 ---
ds = CachedKeywordSpottingDataset(split='test', training=False)
ck = torch.load('models/dscnn_best_cached.pt', map_location='cpu', weights_only=False)
m = DSCNN(); m.load_state_dict(ck['model_state_dict']); m.eval()
sess = ort.InferenceSession('models/dscnn_fp32.onnx', providers=['CPUExecutionProvider'])

pt_all, ox_all = [], []
B = 256
with torch.no_grad():
    for s in range(0, len(ds), B):
        X = torch.stack([ds[i][0] for i in range(s, min(s+B, len(ds)))])
        pt_all.append(torch.sigmoid(m(X)).numpy().squeeze(1))
        ox_all.append(1/(1+np.exp(-sess.run(['logit'], {'log_mel': X.numpy()})[0].squeeze(1))))
pt = np.concatenate(pt_all); ox = np.concatenate(ox_all)

flip = np.where((pt >= 0.85) != (ox >= 0.85))[0]
print('flipped:', len(flip))
for i in flip:
    print(f'  idx={i}: pt={pt[i]:.6f} ox={ox[i]:.6f}')
print('max score diff:', np.abs(pt - ox).max())