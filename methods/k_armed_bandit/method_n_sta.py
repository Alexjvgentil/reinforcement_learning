
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ambiente do Bandido de k-braços (k-Armed Bandit) - Não Estacionário.
# Este ambiente simula k alavancas. Inicialmente q*(a) são iguais (ou aleatórios), mas mudam com o tempo (random walk).
# Ao puxar uma alavanca (ação), o ambiente retorna uma recompensa extraída de uma distribuição Normal(q*(a), 1).
# O objetivo do agente é maximizar a soma das recompensas ao longo do tempo, adaptando-se às mudanças.
class BanditEnvironment:
    def __init__(self, k=10):
        self.k = k
        # True values of q*(a) for each arm. 
        # Usually initialized to 0 or equal values for non-stationary to see the divergence, 
        # or initialized roughly same as stationary. Let's start with 0.
        self.q_true = np.zeros(k) # Starting equal allows us to see how they separate
        # Optimal action
        self.best_action = np.argmax(self.q_true)

    def step(self, action):
        # Reward is N(q*(a), 1)
        reward = np.random.randn() + self.q_true[action]
        
        # Non-stationary dynamics: Random walk
        # q*(a) = q*(a) + N(0, 0.01) for all arms
        self.q_true += np.random.randn(self.k) * 0.01
        
        # Update best action
        self.best_action = np.argmax(self.q_true)
        
        return reward

    def reset(self):
        self.q_true = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)

# Agente Epsilon-Greedy.
# Este algoritmo equilibra exploração e exploração escolhendo a melhor ação estimada com probabilidade 1-epsilon,
# e uma ação aleatória com probabilidade epsilon.
# Parâmetros:
#   k: número de braços.
#   epsilon: probabilidade de exploração (escolha aleatória).
# O agente mantém estimativas Q(a) para cada ação e as atualiza usando uma média amostral ou um passo constante.
class EpsilonGreedyAgent:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.q_estimation = np.zeros(k)
        self.action_counts = np.zeros(k)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            # print("Exploring...")
            return np.random.randint(self.k)
        else:
            # Random tie-breaking is important!
            # np.argmax will pick the first one which introduces bias
            max_val = np.max(self.q_estimation)
            candidates = np.where(self.q_estimation == max_val)[0]
            return np.random.choice(candidates)

    def update(self, action, reward):
        self.action_counts[action] += 1
        # Q(n+1) = Q(n) + (1/n) * (R - Q(n))
        alpha = 1.0 / self.action_counts[action]
        self.q_estimation[action] += alpha * (reward - self.q_estimation[action])

    def reset(self):
        self.q_estimation = np.zeros(self.k)
        self.action_counts = np.zeros(self.k)


# Agente Ganancioso com Inicialização Otimista.
# Este é um método puramente ganancioso (greedy) mas que incentiva a exploração através da inicialização.
# Ao iniciar as estimativas Q(a) com um valor alto (otimista), o agente é forçado a explorar todas as as ações
# no início, pois as recompensas reais serão menores que a estimativa, reduzindo-a gradualmente.
# Parâmetros:
#   alpha: taxa de aprendizado (step-size).
#   initial_value: valor inicial otimista para as estimativas Q(a).
class OptimisticGreedyAgent:
    def __init__(self, k=10, alpha=0.1, initial_value=5.0):
        self.k = k
        self.alpha = alpha
        self.q_estimation = np.full(k, initial_value)
        
    def select_action(self):
        max_val = np.max(self.q_estimation)
        candidates = np.where(self.q_estimation == max_val)[0]
        return np.random.choice(candidates)

    def update(self, action, reward):
        # Constant step-size
        self.q_estimation[action] += self.alpha * (reward - self.q_estimation[action])

    def reset(self):
        # Reset to optimistic initial value
        self.q_estimation = np.full(self.k, 5.0)

# Agente UCB (Upper Confidence Bound).
# Este algoritmo seleciona ações baseando-se no potencial de serem ótimas, considerando tanto a estimativa atual
# quanto a incerteza sobre essa estimativa.
# A fórmula de seleção é: A_t = argmax [ Q_t(a) + c * sqrt(ln(t) / N_t(a)) ]
# Parâmetros:
#   c: parâmetro de exploração que controla o grau de confiança (quanto maior, mais exploração).
# O termo de raiz quadrada é grande para ações pouco exploradas, diminuindo conforme N_t(a) aumenta.
class UCBAgent:
    def __init__(self, k=10, c=2):
        self.k = k
        self.c = c
        self.q_estimation = np.zeros(k)
        self.action_counts = np.zeros(k)
        self.t = 0

    def select_action(self):
        self.t += 1
        # If an arm hasn't been tried, check it first (infinite bound)
        if 0 in self.action_counts:
            candidates = np.where(self.action_counts == 0)[0]
            return np.random.choice(candidates)
        
        uncertainty = self.c * np.sqrt(np.log(self.t) / self.action_counts)
        ucb_values = self.q_estimation + uncertainty
        
        max_val = np.max(ucb_values)
        candidates = np.where(ucb_values == max_val)[0]
        return np.random.choice(candidates)

    def update(self, action, reward):
        self.action_counts[action] += 1
        alpha = 1.0 / self.action_counts[action]
        self.q_estimation[action] += alpha * (reward - self.q_estimation[action])
    
    def reset(self):
        self.q_estimation = np.zeros(self.k)
        self.action_counts = np.zeros(self.k)
        self.t = 0

# Agente Bandit Gradiente.
# Diferente dos outros métodos que estimam valores de ação Q(a), este método aprende uma preferência numérica H(a)
# para cada ação. As preferências não têm interpretação em termos de recompensa.
# As probabilidades de ação são determinadas por uma distribuição Softmax sobre as preferências.
# A atualização das preferências segue uma aproximação de gradiente ascendente estocástico.
# Parâmetros:
#   alpha: taxa de aprendizado.
#   baseline: se True, usa a recompensa média como linha de base para reduzir a variância.
class GradientBanditAgent:
    def __init__(self, k=10, alpha=0.1, baseline=True):
        self.k = k
        self.alpha = alpha
        self.baseline = baseline
        self.preferences = np.zeros(k)
        self.action_probs = np.zeros(k)
        self.average_reward = 0
        self.time_step = 0

    def softmax(self):
        exp_pref = np.exp(self.preferences - np.max(self.preferences)) # Stability
        return exp_pref / np.sum(exp_pref)

    def select_action(self):
        self.action_probs = self.softmax()
        return np.random.choice(self.k, p=self.action_probs)

    def update(self, action, reward):
        self.time_step += 1
        
        # Update baseline (average reward)
        if self.baseline:
            self.average_reward += (1.0/self.time_step) * (reward - self.average_reward)
        
        # Update preferences
        one_hot = np.zeros(self.k)
        one_hot[action] = 1
        
        baseline_term = self.average_reward if self.baseline else 0
        
        # H(a) = H(a) + alpha * (R - R_avg)(1(A=a) - pi(a))
        # This one line handles both selected (1 - pi) and not selected (0 - pi)
        self.preferences += self.alpha * (reward - baseline_term) * (one_hot - self.action_probs)

    def reset(self):
        self.preferences = np.zeros(self.k)
        self.average_reward = 0
        self.time_step = 0

def run_experiment(agent, env, n_steps=1000, n_runs=2000):
    rewards = np.zeros(n_steps)
    optimal_action_counts = np.zeros(n_steps)

    for r in tqdm(range(n_runs)):
        env.reset()
        agent.reset()
        # print(f"Env rewards: {env.q_true}")
        # print(f"Env best action: {env.best_action}")    
        action_history = []
        for t in range(n_steps):
            action = agent.select_action()
            # print(f"Action: {action}")
            # x = input("Press Enter to continue...")
            reward = env.step(action)
            # print(f"Reward: {reward}")
            # x = input("Press Enter to continue...")
            agent.update(action, reward)
            # print(f"Agent updated")
            # x = input("Press Enter to continue...")
            rewards[t] += reward
            
            if action == env.best_action:
                optimal_action_counts[t] += 1
                
            action_history.append(action)
        # print(f"Action history: {action_history}")
        # print(f"Rewards: {rewards}")    
        # x = input("Press Enter to continue...")
    
    avg_rewards = rewards / n_runs
    optimal_action_percentage = (optimal_action_counts / n_runs) * 100
    return avg_rewards, optimal_action_percentage

if __name__ == "__main__":
    k = 10
    n_steps = 1000
    n_runs = 500 # Slightly reduced for speed, can increase if needed

    print(f"Running experiments with k={k}, n_steps={n_steps}, n_runs={n_runs}...")

    env = BanditEnvironment(k=k)

    # 1. Epsilon-Greedy (epsilon=0.1)
    print("Running Epsilon-Greedy...")
    eg_agent_01 = EpsilonGreedyAgent(k=k, epsilon=0.1)
    eg_rewards_01, eg_opt_01 = run_experiment(eg_agent_01, env, n_steps, n_runs)

    # 2. Gradient Bandit (alpha=0.1)
    print("Running Gradient Bandit...")
    gb_agent = GradientBanditAgent(k=k, alpha=0.1)
    gb_rewards, gb_opt = run_experiment(gb_agent, env, n_steps, n_runs)

    # 3. UCB (c=2)
    print("Running UCB...")
    ucb_agent = UCBAgent(k=k, c=2)
    ucb_rewards, ucb_opt = run_experiment(ucb_agent, env, n_steps, n_runs)

    # 4. Optimistic Initialization (alpha=0.1, Q0=5)
    print("Running Optimistic Greedy...")
    op_agent_1 = OptimisticGreedyAgent(k=k, alpha=0.1, initial_value=5)
    op_rewards_1, op_opt_1 = run_experiment(op_agent_1, env, n_steps, n_runs)

    print("Plotting results...")
    plt.figure(figsize=(12, 10))
    
    # Plot Average Reward
    plt.subplot(2, 1, 1)
    plt.plot(eg_rewards_01, label='Epsilon-Greedy (eps=0.1)', color='b', alpha=0.8)
    plt.plot(gb_rewards, label='Gradient Bandit (alpha=0.1)', color='g', alpha=0.8)
    plt.plot(ucb_rewards, label='UCB (c=2)', color='r', alpha=0.8)
    plt.plot(op_rewards_1, label='Optimistic Greedy (Q1=5, alpha=0.1)', color='c', alpha=0.8)
    plt.ylabel('Average Reward')
    plt.title('Performance of K-Armed Bandit Algorithms (Non-Stationary)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot % Optimal Action
    plt.subplot(2, 1, 2)
    plt.plot(eg_opt_01, label='Epsilon-Greedy (eps=0.1)', color='b', alpha=0.8)
    plt.plot(gb_opt, label='Gradient Bandit (alpha=0.1)', color='g', alpha=0.8)
    plt.plot(ucb_opt, label='UCB (c=2)', color='r', alpha=0.8)
    plt.plot(op_opt_1, label='Optimistic Greedy (Q1=5, alpha=0.1)', color='c', alpha=0.8)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = 'bandit_comparison_n_sta.png'
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Results saved to {output_file}")
