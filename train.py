from asteroid.models import ConvTasNet
from asteroid.engine.system import System
from asteroid.engine.optimizers import make_optimizer
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from custom_dataset import CustomMixDataset
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",             # 기준 지표
    dirpath="./model",              # 저장 경로
    filename="best-model-{epoch}",  # 파일 이름
    save_top_k=1,                   # 가장 좋은 모델 하나만 저장
    mode="min"                      # loss는 작을수록 좋으니까 min
)

# 데이터셋 로드
train_dataset = CustomMixDataset("./data", sample_rate=8000)
valid_dataset = CustomMixDataset("./data", sample_rate=8000)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=2)

print("model")
# 모델 정의
model = ConvTasNet(n_src=2)

print("opt")
# 옵티마이저 정의
optimizer = make_optimizer(model.parameters(), lr=1e-3)

print("system")
# 시스템 래핑 (Asteroid용)
system = System(
    model=model,
    optimizer=optimizer,
    loss_func='sisdr',
    train_loader=train_loader,    # 필수 인자 추가!
    val_loader=valid_loader       # 검증 데이터도 함께 넣는 게 일반적
)

print("train")
# PyTorch Lightning Trainer 사용
trainer = Trainer(max_epochs=30, accelerator="auto", devices="auto")
trainer.fit(system, train_dataloaders=train_loader, val_dataloaders=valid_loader)
