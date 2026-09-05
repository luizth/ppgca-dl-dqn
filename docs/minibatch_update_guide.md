# Casos comuns de erro no minibatch update

## O erro comum

A seguinte função parece que faz atualização em minibatch, e depois
calcula a média do erro para apresentar o resultado.

`agent.py:update_Q_network`
```python
def update_Q_network(batch_size=32):
    losses = []

    minibatch = self.eb.sample(self.batch_size)

    for exp in minibatch:
        action = exp.action
        reward = exp.reward
        done = exp.done

        # Those are tensors
        state_features = exp.state
        next_state_features = exp.next_state

        # Calculate TD target - bootstrap
        with torch.no_grad(): # No need to track gradients for target calculation
            if done:
                td_target = torch.tensor(reward, dtype=torch.float32)
            else:
                next_q_values = self._get_Q_target_values(next_state_features)
                td_target = torch.tensor(reward, dtype=torch.float32) + self.discount_factor * torch.max(next_q_values)

        # Get the current Q-value prediction
        q_values = self._get_Q_values(state_features)
        q_value = q_values[action]

        # Compute loss
        loss = self.lossfn(td_target, q_value)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store loss
        losses.append(loss.item())

    return losses
```

mas há um erro grave.

- `update_Q_network` não faz atualização em minibatch.

O laço, na verdade, faz um backward() + optimizer.step() por amostra,
32 passos SGD sequenciais por passo do ambiente. As amostras são avaliadas
contra uma rede já deslocada pelas anteriores, não é o gradiente do minibatch.

## O que a função faz

A função faz 32 (tamanho do minibatch) atualizações **online** sequenciais.
As amostras **não** vem da mesma rede, porque em cada passo a rede é atualizada,
então o `Q_target` muda a cada passo, e o Q muda a cada passo.

A seguir, demonstramos quanto os pesos da rede andaram no caso implementado acima
em comparação com o minibatch real:

Modo            | \|\|dW\|\|
--              | --
sequencial      | 0.33215
minibatch real  | 0.09825

O cosseno entre as direções é **0.8287** (1.0 seria mesma direção). A análise
completa está na última seção do arquivo.

## Como implementar minibatch

O minibatch é gerenciado automaticamente utilizando `torch.stack`.

```python
def update_Q_network(batch_size=32):
    minibatch = self.eb.sample(self.batch_size)

    # Empilhar o minibatch em tensores para processamento em lote
    # torch.stack will create a tensor of shape (batch_size [32], 1, 4, 84, 84)
    states      = torch.stack([e.state for e in minibatch])
    next_states = torch.stack([e.next_state for e in minibatch])
    actions     = torch.tensor([e.action for e in minibatch])
    rewards     = torch.tensor([e.reward for e in minibatch], dtype=torch.float32)
    dones       = torch.tensor([e.done for e in minibatch], dtype=torch.float32)

    # DQN - um passo de gradiente sobre a média do minibatch
    with torch.no_grad():
        targets = rewards + self.discount_factor * self.Q_target(next_states).max(1).values * (1 - dones)

    q = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = self.lossfn(q, targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.item()
```

Desse modo, todas as experiências são agrupadas e paralelizadas. Apenas
um passo é dado utilizando a média das experiências, produzindo um único
sinal de `loss`.

## Análise

### `dW` = o quanto os pesos andaram.

Pense na rede como um ponto num mapa. Treinar é dar passos com esse ponto.
dW é a seta de onde ele estava até onde ele foi parar depois de um passo do ambiente.
||dW|| é só o comprimento dessa seta: a distância percorrida.

O cosseno = se as duas setas apontam para o mesmo lado.
- 1.0 → mesmíssima direção
- 0.0 → uma para o norte, outra para o leste
- -1.0 → direções opostas

Deu 0.83: apontam mais ou menos para o mesmo lado, mas torto.
Então o laço não erra só na distância, **ele vai para um lugar um pouco diferente.**

### Uma analogia pode nos ajudar a entender (O 3,4x)

32 amigos te dão conselho sobre para onde andar.
- Jeito certo (minibatch): você ouve os 32, tira a média, dá um passo.
- Jeito do código: você obedece o amigo 1 por inteiro, anda. Aí ouve o amigo 2,
do lugar novo, e anda. E assim por diante, 32 vezes.

No fim você parou 3,4x mais longe, e num ponto meio torto.

### E por que não dá para dizer "é só um learning rate 32x maior"?

Porque cada passo muda o chão do próximo.
O amigo 5 dá um conselho diferente do que daria se você não tivesse andado antes.
Então o total não é 32 x nada, depende de quais amigos calharam de estar no grupo
e de onde a rede está naquele momento. Hoje deu 3,4x, amanhã dá outro número.

É esse o problema: o tamanho do passo vira uma variável escondida
que ninguém controla nem consegue prever.
