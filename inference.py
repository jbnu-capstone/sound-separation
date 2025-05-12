from asteroid.models import ConvTasNet
import torchaudio
import torch

# 모델 로드 (학습된 모델 ckpt)
model_name = input('Model Name: ')
model = ConvTasNet.from_pretrained("./model/" + model_name + ".ckpt")
model.eval()

# 오디오 파일 로딩
input_wav, sr = torchaudio.load("/path/to/noisy_mix.wav")
assert sr == model.sample_rate, f"Sample rate mismatch: model {model.sample_rate}, input {sr}"

# 배치 차원 추가
input_tensor = input_wav.unsqueeze(0)

# 추론
with torch.no_grad():
    est_sources = model(input_tensor)  # shape: [1, n_src, T]

# 저장
for i, src in enumerate(est_sources[0]):
    torchaudio.save(f"separated_source_{i}.wav", src.unsqueeze(0), sr)
