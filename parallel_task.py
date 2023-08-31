

from concurrent.futures import ThreadPoolExecutor


class ParallelTask():
  def __init__(self) -> None:
    self.executor = ThreadPoolExecutor()

  def inference(self, ids, agent_input, reward_generator, infer_net, RND_net, embedding_net):
    future_infer = self.executor.submit(infer_net, agent_input)
    reward_input = agent_input.state.squeeze(1)
    future_random = self.executor.submit(RND_net.random, reward_input)
    future_predict = self.executor.submit(RND_net.predict, reward_input)
    future_embedding = self.executor.submit(embedding_net, reward_input)

    embedding = future_embedding.result()
    rnd_loss = RND_net.get_loss(future_random.result(), future_predict.result())

    future_reward = self.executor.submit(
      reward_generator.get_intrinsic_reward,
      ids,
      rnd_loss.detach().cpu().numpy(),
      embedding.detach().cpu().numpy()
    )

    return future_infer.result(), future_reward

  def train(self, transitions, RND_net, embedding_net):
    self.executor.submit(RND_net.train, transitions)
    self.executor.submit(embedding_net.train, transitions)

  def get_agent_output(self, transitions, agent):
    future_online_output = self.executor.submit(agent.get_online_output, transitions)
    future_target_output = self.executor.submit(agent.get_target_output, transitions)

    return future_online_output, future_target_output
