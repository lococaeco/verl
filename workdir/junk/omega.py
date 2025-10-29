from omegaconf import OmegaConf

cfg = OmegaConf.load("example.yaml")
print(cfg.trainer.total_epochs)  
print(cfg.trainer.batch_size)  
print(cfg.model.name)              
print(cfg.model.lr)              


"""
Hydra는 “딥러닝 실험을 관리하는 설정 프레임워크”입니다.
즉, 하나의 Python 엔트리포인트(main)에서
다양한 YAML 설정 조합을 자동으로 로드·병합·추적해줍니다.
"""

"""
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)
"""

"""
Hydra가 config/ppo_trainer.yaml을 자동으로 로드함
CLI 인자(data.train_files=...)를 동적으로 override
모든 걸 OmegaConf 객체(config) 로 main()에 넘겨줌

python -m verl.trainer.main_ppo data.train_batch_size=1024 <- 이렇게 override 됨
"""