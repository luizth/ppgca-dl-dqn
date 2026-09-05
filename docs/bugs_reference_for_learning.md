# Bugs: Descrição e como resolver

[x] 1. A rede alvo nunca é sincronizada — src/agent.py:217

```python
if self._number_of_steps_taken_in_episode % self.C == 0:
```

O contador é zerado a cada episódio (agent.py:208), e o return antecipado em agent.py:209 pula esse bloco justamente no passo terminal. Com C=100, qualquer episódio mais curto que 100 passos nunca dispara a sincronização. Rodei 600 passos de CartPole (28 episódios): Q_target continuava bit a bit idêntica à inicialização aleatória. O contador precisa ser global, não por episódio.

[x] 2. truncated é descartado no desempacotamento — src/agent.py:183 e :242

```python
next_state, reward, done, info, _ = self._env.step(action)
```

Gymnasium retorna (obs, reward, terminated, truncated, info). Então done = terminated, info = truncated e _ = o info real. Confirmado em execução. Consequências:
- MountainCar-v0: o limite de 200 passos é ignorado; num seed o episódio foi até 2826 passos antes de terminar de verdade, e em outro não terminou em 3000.
- CliffWalking-v1 não tem `max_episode_steps` nenhum.
- CartPole: um agente que aprende a equilibrar nunca aciona terminated → while not done em main.py:168 não sai.

Para o bootstrap o done está certo (não se deve bootstrapar em terminated), mas o fim de episódio precisa ser terminated or truncated.

[x] 3. ContinuousNormalized.encode destrói a observação — `src/feature_construction.py:27`

```python
normalized = (state - np.min(state)) / (np.max(state) - np.min(state))
```

Normaliza min-max entre as dimensões do próprio vetor, misturando grandezas físicas diferentes. Em MountainCar (2 dimensões) isso é matematicamente degenerado — o resultado só pode ser [0,1] ou [1,0]. Verifiquei 300 passos: o vetor codificado foi [0., 1.] em todos eles. O agente não consegue distinguir estado algum. Em CartPole o mapeamento é descontínuo e não estacionário (o mesmo valor bruto vira 0.69 ou 1.0 conforme qual componente for o máximo). E max == min dá divisão por zero → [nan nan nan nan].

[x] 4. Inicialização de pesos uniform(-1, 1) — src/network.py:11,15

Para o MLP pequeno ainda passa, mas na rede convolucional os Q-values iniciais saem na casa de 2×10⁵ ([16861, 202514, 9054, -95345]), contra ~0.03 com a inicialização padrão do PyTorch. Diverge no primeiro passo. Vale notar que DQN.__init__ (network.py:97) ainda reaplica esse init sobre a CNN e o MLP já inicializados.

[x] 5. NameError no caminho convolucional — src/main.py:111

in_channels=m é usado na linha 111, mas m = 4 só é atribuído na linha 126. Reproduzido: NameError: name 'm' is not defined. O branch use_conv=True nunca rodou.

Bugs que degradam o treino

[x] 6. update_Q_network não faz atualização em minibatch — agent.py:135-165

O laço faz um backward() + optimizer.step() por amostra, 32 passos SGD sequenciais por passo do ambiente. As amostras 2..32 são avaliadas contra uma rede já deslocada pelas anteriores — não é o gradiente do minibatch. Medi: 31,17 ms/passo contra 1,64 ms vetorizado (19× mais lento).

[x] 7. O preprocessador é alimentado duas vezes por passo e nunca é resetado — agent.py:174,190-191

get_state_tensor tem efeito colateral (empurra no frame_buffer e no state_buffer). Chamá-lo para state e de novo para next_state insere 2 frames por passo do ambiente. Traço real das pilhas:

env step 1: state=[1,1,1,1]  next_state=[1,1,1,2]
env step 2: state=[1,1,2,2]  next_state=[1,2,2,3]
env step 3: state=[2,2,3,3]  next_state=[2,3,3,4]

Cada frame aparece duplicado e o max-pooling de 2 frames acaba operando sobre frames repetidos. Além disso preprocessor.reset() nunca é chamado no reset do episódio — o primeiro estado do episódio novo ficou [4, 4, 5, 99], com 3 frames do episódio anterior.

[x] 8. O decay do epsilon é pulado no passo terminal — agent.py:205-209. O return antecipado pula tanto o decay quanto a atualização da target.

[x] 9. reset() não limpa o replay buffer nem o preprocessador — agent.py:87-95. Reseta rede, otimizador e epsilon, mas o self.eb guarda transições geradas pela rede antiga.

[ ] 10. Sem warm-up do replay — treina já no primeiro passo, com min(batch_size, len(buffer)) devolvendo um minibatch de 1 amostra.

[x] 11. Clipping de recompensa por sinal aplicado antes de logar — agent.py:186 + main.py:173. Em LunarLander o +100 do pouso vira +1, igual a um tick de shaping — o sinal da tarefa desaparece. Em CartPole com sutton_barto_reward=True as recompensas já são só {0, -1}, então episode_reward é a constante −1 em todo episódio. Se o clipping é intencional (é o do paper), ele deveria ser aplicado só no alvo de treino, e a métrica logada deveria usar a recompensa bruta.

Código morto com bugs

[x] 12. DeepQLearning.train() (agent.py:222-259) — import tqdm (agent.py:2) importa o módulo, então tqdm(range(...)) levanta TypeError: 'module' object is not callable. Confirmado. Esse caminho também ignora o clipping, ignora o preprocessador e repete o desempacotamento errado do step.

[x] 13. baseline.QLearning.train() — usa self.eb (baseline.py:100), atributo que a classe nunca define → AttributeError. E step_rewards[step] (baseline.py:96) é uma expressão sem atribuição: o array volta zerado.

[x] 14. JobConfig.name (config.py:12) — o default str(uuid.uuid4())[:8] é avaliado uma vez na importação da classe, então seria compartilhado por todos os jobs. Latente hoje porque todo config passa name explicitamente.

Menores

- [x] eb.py:38 — np.random.choice(self.buffer, ...) converte a lista inteira (10 000 itens) num array de objetos a cada passo. Custo O(n) por passo; use random.sample.
- [] pre_processing.py:56 — gray_frame = max_frame[1] pega só o canal G, não a luminância que a docstring promete.
- [x] agent.py:113,119 — choose_action monta grafo de autograd que é descartado; falta torch.no_grad().
- [] main.py:143 — number_of_states=in_size para espaços Box faz o QLearning.__init__ alocar uma Q-table 4×2 inútil, sobrescrita logo em seguida.
- [] main.py:206-209 — run_job não retorna nada, então results.txt recebe uma linha None por job. Nenhum env.close(), nenhuma semente fixada.
- [x] agent.py:157 — self.lossfn(td_target, q_value) está com os argumentos invertidos em relação à convenção (input, target), mas não é bug: verifiquei que o gradiente é idêntico (−6.0 nos dois casos), já que o MSE é simétrico e o autograd propaga pelo argumento target.
