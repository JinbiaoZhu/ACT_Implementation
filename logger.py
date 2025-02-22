import wandb
import torch


class WandbLogger:
    def __init__(self, project_name, config=None):
        """
        初始化 wandb，设置项目名称和其他配置
        :param project_name: 项目名称
        :param entity: 用户/团队名称，默认为 None（可以设置为你的 wandb 实体）
        :param config: 一个字典，包含一些超参数或配置
        """
        self.project_name = project_name
        self.config = config if config else {}

        # 初始化 wandb
        wandb.init(project=self.project_name, config=self.config)

    def log(self, metrics, step=None):
        """
        记录当前的训练/验证指标
        :param metrics: 字典形式的指标，如 {'loss': 0.5, 'accuracy': 0.9}
        :param step: 训练的步数（可选）
        """
        wandb.log(metrics, step=step)

    def save_model(self, model, model_name="model.h5"):
        """
        保存模型到 wandb 服务器
        :param model: 需要保存的模型
        :param model_name: 模型文件的名称
        """
        # 保存模型到本地文件
        torch.save(model.state_dict(), model_name)
        # wandb.save(model_name)
        print(f"Model saved as {model_name} in local")

    def finish(self):
        """结束 wandb 运行"""
        wandb.finish()
