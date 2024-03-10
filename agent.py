
import os
import torch
from data_type import DataType
from model import DecisionTransformer, Input
import torch.nn.functional as F

from target_manager import TargetManager


class Agent:
  def __init__(self, config) -> None:
    self._data_type = DataType(config)
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    self.device = device

    self.net = DecisionTransformer(config, device).to(device)
    self.criteria = torch.nn.CrossEntropyLoss(reduction="none")
    self.opt = torch.optim.Adam(
        params=self.net.parameters(),
        lr=config.adam_lr,
        betas=config.adam_beta,
    )

    self._config = config
    self.env = TargetManager(config).env(config.env_name, config.max_timestep)
    self.train_steps = 0

    self.load()

  def load(self):
    if os.path.exists(self._config.checkpoint_path):
      checkpoint = torch.load(self._config.checkpoint_path)
      self.net.load_state_dict(checkpoint["model"])
      self.opt.load_state_dict(checkpoint["optimizer"])

  def save(self):
    checkpoint = {"model": self.net.cpu().state_dict(),
                  "optimizer": self.opt.state_dict()}
    torch.save(checkpoint, self._config.checkpoint_path)
    self.net.to(self.device)

  def train(self, input):
    self.net.train()

    # with torch.autocast(device_type=self._data_type, dtype=torch.float16, enabled=self._config.use_amp):
    logits = self.net(input)
    targets = input.action.to(torch.long)

    loss = self.criteria(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss = loss.mean()

    loss.backward()

    # Since the gradients of optimizer's assigned parameters are now unscaled, clips as usual.
    # You may use the same value for max_norm here as you would without gradient scaling.
    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self._config.grad_norm_clip)

    self.opt.step()
    self.opt.zero_grad()

    self.train_steps += 1

    if self.train_steps % 10 == 0:
      self.save()

    return loss

  def sample_action(self, input):
    with torch.no_grad():
      logits = self.net(input)
    probs = F.softmax(logits[:, -1, :], dim=-1)
    action = torch.multinomial(probs, num_samples=1).item()
    return action

  def eval(self):
    self.net.eval()

    env_output = self.env.reset()

    states, actions, rtgs = [env_output.next_state], [0], [env_output.reward]
    total_reward = 0
    steps = 0

    for i in range(self._config.max_timestep):
      input = Input(
        rtg=torch.tensor(rtgs),
        state=torch.tensor(states),
        action=torch.tensor(actions),
        timestep=torch.tensor(steps),
      )

      sampled_action = self.sample_action(input)
      env_output = self.env.step(sampled_action)

      states += env_output.next_state
      rtgs += [rtgs[-1] - env_output.reward]
      actions += [sampled_action]

      total_reward += env_output.reward
      steps += 1
      if env_output.done:
        break

    return total_reward
