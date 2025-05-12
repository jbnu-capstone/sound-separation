import sounddevice as sd
import torchaudio
import torch
from asteroid.models import ConvTasNet
import numpy as np
import queue
import threading

# 모델 로드
model_name = input('Model Name: ')
model = ConvTasNet.from_pretrained("./model/" + model_name + ".ckpt")
model.eval()

# 설정
sample_rate = 16000
block_duration = 1.0  # 1초 단위로 추론
block_size = int(sample_rate * block_duration)

# 실시간 오디오 버퍼
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    """ 마이크 입력을 큐에 넣음 """
    if status:
        print(status)
    q.put(indata.copy())


def inference_thread():
    """ 추론 스레드 """
    while True:
        if not q.empty():
            block = q.get()
            block_tensor = torch.from_numpy(block.T).float()  # shape: [1, T]
            with torch.no_grad():
                est_sources = model(block_tensor.unsqueeze(0))  # [1, n_src, T]
            est_sources = est_sources.squeeze(0).numpy()  # [n_src, T]

            # 파일 저장
            for i, src in enumerate(est_sources):
                filename = f"output_source_{i}_{np.random.randint(1000)}.wav"
                torchaudio.save(filename, torch.from_numpy(src).unsqueeze(0), sample_rate)
                print(f"Saved: {filename}")


def main():
    stream = sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        blocksize=block_size,
        callback=audio_callback
    )
    thread = threading.Thread(target=inference_thread, daemon=True)
    thread.start()

    print("🎙️ 실시간 추론 시작... Ctrl+C로 종료")
    with stream:
        while True:
            sd.sleep(100)


if __name__ == "__main__":
    main()
