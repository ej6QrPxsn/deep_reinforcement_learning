
import os
import numpy as np
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
    self.opt = self.configure_optimizers(config)

    self._config = config
    self.env = TargetManager(config).env(config.env_name, config.max_timestep)
    self.train_steps = 0

    self.load()

  def configure_optimizers(self, config):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    # whitelist_weight_modules = (torch.nn.Linear, )
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in self.net.named_modules():
      for pn, p in m.named_parameters():
        fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

        if pn.endswith('bias'):
          # all biases will not be decayed
          no_decay.add(fpn)
        elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
          # weights of whitelist modules will be weight decayed
          decay.add(fpn)
        elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
          # weights of blacklist modules will NOT be weight decayed
          no_decay.add(fpn)

    # special case the position embedding parameter in the root GPT module as not decayed
    no_decay.add('pos_emb')
    no_decay.add('global_pos_emb')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in self.net.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=config.adam_lr, betas=config.adam_beta)
    return optimizer

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
        rtg=torch.tensor(np.array(rtgs)[-self._config.context_length:]).reshape(1, -1, 1).to(torch.float32).to(self.device),
        state=torch.tensor(np.array(states)[-self._config.context_length:]).unsqueeze(0).to(torch.float32).to(self.device),
        action=torch.tensor(np.array(actions)[-self._config.context_length:]).reshape(1, -1, 1).to(torch.int64).to(self.device),
        timestep=torch.tensor(np.array(steps)).to(torch.int64).reshape(1, 1, 1).to(self.device),
      )

      sampled_action = self.sample_action(input)
      env_output = self.env.step(sampled_action)

      states += [env_output.next_state]
      rtgs += [rtgs[-1] - env_output.reward]
      actions += [sampled_action]

      total_reward += env_output.reward
      steps += 1
      if env_output.done:
        break

    return total_reward
