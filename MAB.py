import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# class to model ad behavior
class BernoulliBandit(object):
  def __init__(self, p):
    self.p = p
    
  def display_banner(self):
    reward = np.random.binomial(n = 1, p = self.p)
    return reward
  

# arbitrarly pick CTR and create ad banners

adA = BernoulliBandit(0.005)
adB = BernoulliBandit(0.018)
adC = BernoulliBandit(0.025)
adD = BernoulliBandit(0.029)
adE = BernoulliBandit(0.032)

ads = [adA, adB, adC, adD, adE]

df_reward_comparison = pd.DataFrame()

# A/B/n testing

# create variables to keep track of rewards in the experiment

n_test = 10000
steps = 90000

n_ads = len(ads)

Q = np.zeros(n_ads)   # action values
N = np.zeros(n_ads)  
cumulative_reward = 0
avg_rewards = []    # average rewards over time
optimal_action_total = 0 
optimal_action_pct = [] 

# run A/B/n test

for i in range(n_test):
  action_taken = np.random.randint(n_ads)
  R = ads[action_taken].display_banner()   # Observe the reward
  N[action_taken] += 1
  Q[action_taken] += ( 1 / N[action_taken]) * ( R - Q[action_taken])
  cumulative_reward += R
  avg_cumulative_reward = cumulative_reward / ( i + 1)
  avg_rewards.append(avg_cumulative_reward)
  
  optimal_ad_index = np.argmax(Q)
  optimal_action_total += (action_taken == optimal_ad_index)
  optimal_action_pt = optimal_action_total/(i+1) 
  optimal_action_pct.append(optimal_action_pt)
  
best_ad_index = np.argmax(Q)

print("The best performing bandit is {}".format(chr(ord('A') + best_ad_index)))

# On the test set, A/B/n test has identified the correct ad.

# Now, let's  run this model in production

action_taken = best_ad_index
for i in range(steps):
  R = ads[action_taken].display_banner()
  cumulative_reward += R
  avg_cumulative_reward = cumulative_reward / (n_test + i + 1)
  avg_rewards.append(avg_cumulative_reward)
  

# create dataframe to record results
df_reward_comparision = pd.DataFrame(avg_rewards, columns=['A/B/n'])

plt.figure(figsize=(10, 6))
plt.plot(df_reward_comparision['A/B/n'], label='A/B/n')
plt.title(f"A/B/n Test Average Reward: {avg_cumulative_reward:.4f}")
plt.xlabel('Impressions')
plt.ylabel('Avg. Reward')
plt.legend()
plt.grid(True)
plt.savefig('reward_comparison_plot.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(optimal_action_pct, label='Optimal Action Percentage (A/B/n)')
plt.title("Optimal Action Percentage over Time (A/B/n)")
plt.xlabel('Trials')
plt.ylabel('Optimal Action Percentage')
plt.legend()
plt.grid(True)
plt.savefig('optimal_action_plot.png')
plt.show()



# epsilon-greedy actions

# Define epsilon values , 0 is greedy and others are eps-greedy
eps_values = [0, 0.05, 0.1, 0.2]
steps = 100000
n_ads = len(ads)
optimal_action_pct_dict = {eps: [] for eps in eps_values}

# Run experiment for each epsilon value
for eps in eps_values:
  Q = np.zeros(n_ads)
  N = np.zeros(n_ads)
  cumulative_reward = 0
  avg_rewards = []
  optimal_action_total = 0 
  optimal_action_pct = [] 

  action_taken = np.random.randint(n_ads)

  for i in range(steps):
    R = ads[action_taken].display_banner()
    N[action_taken] += 1
    Q[action_taken] += (1/N[action_taken]) * (R - Q[action_taken])
    cumulative_reward += R
    avg_cumulative_reward = cumulative_reward / (i + 1)
    avg_rewards.append(avg_cumulative_reward)

    if np.random.uniform() <= eps:
      action_taken = np.random.randint(n_ads)
    else:
      action_taken = np.argmax(Q)
      
    optimal_action = np.argmax([b.p for b in ads]) 
    optimal_action_total += (action_taken == optimal_action) 
    optimal_action_pt = optimal_action_total/(i+1) 
    optimal_action_pct.append(optimal_action_pt) 

  df_reward_comparision[f'e-greedy: {eps}'] = avg_rewards
  optimal_action_pct_dict[eps] = optimal_action_pct


plt.figure(figsize=(10, 6))
for eps in eps_values:
  plt.plot(df_reward_comparision.index, df_reward_comparision[f'e-greedy: {eps}'], label=f'e-greedy: {eps}')
plt.title("e-greedy actions")
plt.xlabel("Impressions")
plt.ylabel("Average Reward")
plt.legend()
plt.grid(True)
plt.savefig('reward_comparison_plot.png')
plt.show()

plt.figure(figsize=(10, 6))
for eps in eps_values:
  plt.plot(range(1, steps + 1), optimal_action_pct_dict[eps], label=f'e-greedy: {eps}')
plt.title("Optimal Action Percentage")
plt.xlabel("Impressions")
plt.ylabel("Optimal Action Percentage")
plt.legend()
plt.grid(True)
plt.savefig('optimal_action_percentage_plot.png')
plt.show()


# Upper Confidence Bound

c_values = [2**(1/20), 0.05, 0.1, 0.2]
steps = 100000
n_ads = len(ads)
ad_indices = np.array(range(n_ads))


for c in c_values:
  Q = np.zeros(n_ads)
  N = np.zeros(n_ads)
  cumulative_reward = 0
  avg_rewards = []
  optimal_action_total = 0
  optimal_action_pct = []
  
  # Run experiment
  for t in range(1, steps + 1):
    if any(N == 0):
      action_taken = np.random.choice(ad_indices[N == 0])
    else:
      uncertainty = np.sqrt(np.log(t) / N)
      action_taken = np.argmax(Q + c * uncertainty)
  
    R = ads[action_taken].display_banner()
    N[action_taken] += 1
    Q[action_taken] += (1 / N[action_taken]) * (R - Q[action_taken])
    cumulative_reward += R
    avg_cumulative_reward = cumulative_reward / t
    avg_rewards.append(avg_cumulative_reward)
  
    optimal_ad_index = np.argmax(Q)
    optimal_action_total += (action_taken == optimal_ad_index)
    optimal_action_pt = optimal_action_total / t
    optimal_action_pct.append(optimal_action_pt)
  
  df_reward_comparison[f'UCB: {c}'] = avg_rewards
  df_reward_comparison[f'Optimal Action Percentage (UCB: {c})'] = optimal_action_pct
  print(f"Optimal Action Percentage for UCB={c}: {optimal_action_pct[-1]:.4f}")

best_reward = df_reward_comparison.loc[t - 1, [f'UCB: {c}' for c in c_values]].max()
print(f"The best reward at time {t} is: {best_reward}")

# Plotting Average Reward
plt.figure(figsize=(10, 6))
for c in c_values:
  plt.plot(df_reward_comparison.index, df_reward_comparison[f'UCB: {c}'], label=f'UCB: {c}')
plt.title("UCB actions. Best Average Reward: {:.4f}".format(best_reward))
plt.xlabel("Impressions")
plt.ylabel("Average Reward")
plt.legend()
plt.grid(True)
plt.savefig('reward_comparison_plot.png')
plt.show()

# Plotting Optimal Action Percentage
plt.figure(figsize=(10, 6))
for c in c_values:
  plt.plot(df_reward_comparison.index, df_reward_comparison[f'Optimal Action Percentage (UCB: {c})'], label=f'UCB: {c}')
plt.title("Optimal Action Percentage over Time (UCB)")
plt.xlabel("Trials")
plt.ylabel("Optimal Action Percentage")
plt.legend()
plt.grid(True)
plt.savefig('optimal_action_percentage_plot.png')
plt.show()


# Thompson Sampling

steps = 100000
n_ads = len(ads)
alphas = np.ones(n_ads)
betas = np.ones(n_ads)
cumulative_reward = 0
avg_rewards = []
optimal_action_total = 0
optimal_action_pct = []

# run experiment

for i in range(0, steps):
  theta_samples = [np.random.beta(alphas[k], betas[k]) for k in range(n_ads)]
  action_taken = np.argmax(theta_samples)
  R = ads[action_taken].display_banner()
  alphas[action_taken] += R
  betas[action_taken] += 1 - R
  cumulative_reward += R
  avg_cumulative_reward = cumulative_reward / (i + 1)
  avg_rewards.append(avg_cumulative_reward)

  optimal_ad_index = np.argmax([np.random.beta(alphas[k], betas[k]) for k in range(n_ads)])
  optimal_action_total += (action_taken == optimal_ad_index)
  optimal_action_pt = optimal_action_total / (i + 1)
  optimal_action_pct.append(optimal_action_pt)

# Plotting Average Reward
plt.figure(figsize=(10, 6))
plt.plot(avg_rewards, label='Thompson Sampling')
plt.title("Thompson Sampling - Average Reward")
plt.xlabel("Impressions")
plt.ylabel("Average Reward")
plt.legend()
plt.grid(True)
plt.savefig('thompson_sampling_reward_plot.png')
plt.show()

# Plotting Optimal Action Percentage
plt.figure(figsize=(10, 6))
plt.plot(optimal_action_pct, label='Optimal Action Percentage (Thompson Sampling)')
plt.title("Thompson Sampling - Optimal Action Percentage")
plt.xlabel("Trials")
plt.ylabel("Optimal Action Percentage")
plt.legend()
plt.grid(True)
plt.savefig('thompson_sampling_optimal_action_percentage_plot.png')
plt.show()


# Gradient Bandit Algorithm

# Gradient Bandit Algorithm with baseline
steps = 100000
alpha = 0.05
n_ads = len(ads)
H = np.zeros(n_ads)
P = np.ones(n_ads) / n_ads  # equal probability of being selected
avg_reward = 0
optimal_action_total = 0
optimal_action_pct = []

for i in range(1, steps + 1):
  action_taken = np.random.choice(n_ads, size=1, p=P)[0]
  R = ads[action_taken].display_banner()  # observe reward
  avg_reward += (1 / i) * (R - avg_reward)

  # Update action preferences
  for a in range(n_ads):
    if a == action_taken:
      H[a] += alpha * (R - avg_reward) * (1 - P[a])
    else:
      H[a] += -alpha * (R - avg_reward) * P[a]

  # Update action probabilities
  P = np.exp(H - np.max(H)) / np.sum(np.exp(H - np.max(H))) + 1e-5

  # Calculate optimal action percentage
  optimal_ad_index = np.argmax(P)
  optimal_action_total += (action_taken == optimal_ad_index)
  optimal_action_pt = optimal_action_total / i
  optimal_action_pct.append(optimal_action_pt)

# Gradient Bandit Algorithm without baseline
steps = 100000
H = np.zeros(n_ads)
P = np.ones(n_ads) / n_ads  # equal probability of being selected
avg_reward = 0
optimal_action_total = 0
optimal_action_pct_without_baseline = []

for i in range(1, steps + 1):
  action_taken = np.random.choice(n_ads, size=1, p=P)[0]
  R = ads[action_taken].display_banner()  # observe reward
  avg_reward += 0

  # Update action preferences
  for a in range(n_ads):
    if a == action_taken:
      H[a] += alpha * (R - avg_reward) * (1 - P[a])
    else:
      H[a] += -alpha * (R - avg_reward) * P[a]

  # Update action probabilities
  P = np.exp(H - np.max(H)) / np.sum(np.exp(H - np.max(H))) + 1e-5

  # Calculate optimal action percentage
  optimal_ad_index = np.argmax(P)
  optimal_action_total += (action_taken == optimal_ad_index)
  optimal_action_pt = optimal_action_total / i
  optimal_action_pct_without_baseline.append(optimal_action_pt)

# Plotting Optimal Action Percentage for both versions
plt.figure(figsize=(10, 6))
plt.plot(optimal_action_pct, label='With Baseline')
plt.plot(optimal_action_pct_without_baseline, label='Without Baseline')
plt.title("Gradient Bandit Algorithm - Optimal Action Percentage")
plt.xlabel("Trials")
plt.ylabel("Optimal Action Percentage")
plt.legend()
plt.grid(True)
plt.savefig('gradient_bandit_optimal_action_percentage_plot.png')
plt.show()

