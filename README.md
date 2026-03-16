#   Flappy Bird Reinforcement Learning Agent
## SARSA vs. Q-Learning

# ![flappybird_1](https://github.com/Efi-Pecani/FlappyBird-RL-Agent/blob/main/Flappy_Bird_Logo.png)

# ![flappybird_2](https://github.com/Efi-Pecani/FlappyBird-RL-Agent/blob/main/flappybird_phisics.png)

#


# Introduction

Our  objective is to develop a reinforcement learning agent capable of solving the Flappy Bird game. The primary goal was to train an agent that could successfully pass through 10 or more pipes in at least 90% of its trials.

Our approach involved a thorough understanding of the environment to design an effective preprocessing function that reduces the high-dimensional observation space into a manageable state representation. Additionally, we focused on engineering an optimal reward function, balancing the agent's exploration of new states with the exploitation of known rewards, and fine-tuning hyperparameters to ensure the agent converges within a reasonable number of episodes while achieving the desired success rate.

To solve the problem, we implemented and evaluated two reinforcement learning algorithms: Q-Learning and SARSA. These algorithms were chosen for their complementary strengths, enabling us to explore the trade-offs between off-policy and on-policy learning in this dynamic and challenging environment.

# Methodology

## 

## **Preprocessing**

As mentioned, the environment of the Flappy Bird game is complex, and the observation space is high-dimensional causing a large Q-table. To enable the agent to efficiently interact with and learn from the environment, preprocessing was used to simplify the observation space.

The raw observation map of the game contains eight features, some of them has a max value of 512 and a total of \[512 x 20 x 288 x 512 x 512 x 288 x 512 x 512\] states.

First, we reduced the number of features in the observation map. Some features didn’t provide any new information or may be combined with other features to give us information.

* ”next\_next\_pipe\_dist\_to\_player” is always 144 more than ”next\_pipe\_dist\_to\_player” so it doesn’t give us any new information and was removed.  
* ”next\_pipe\_top\_y” and  ”next\_pipe\_bottom\_y” were combined to compute the center of the next pipe, this simplification eliminates unnecessary details while focusing on the spatial relationships crucial for navigation  
* ”next\_next\_pipe\_top\_y” and ”next\_next\_pipe\_bottom\_y” were combined to computed the center of the next-next pipe, this simplification eliminates unnecessary details while focusing on the spatial relationships crucial for navigation  
* The vertical position of the player relative to the next pipe center and the next-next pipe center was calculated using “player\_y” feature.

After the first analysis, we remained with 4 features and a total of \[512 x 20 x 288 x 512\] states.

Then we figured out that the vertical distance to the next-next pipe was irrelevant to immediate decision making so it was removed as well to reduce the dimensions of the state space.

Finally, we decided to apply quantization to discretize the continuous variables into manageable ranges, reducing the size of the state space while retaining critical information.

* Vertical Position to Pipe Center (129 values): The difference between the player's y-position and the pipe center was divided by 4, mapping it to 129 discrete values. This choice balances granularity and computational efficiency, allowing the agent to distinguish between different states effectively.  
* Player Velocity (Unchanged): The player's velocity was kept as is, as it already falls within a manageable range. This feature is essential for understanding the dynamics of the game, such as gravity and acceleration.  
* Horizontal Distance to Next Pipe (37 values): The horizontal distance to the next pipe was divided by 8 and clipped to a maximum of 36 values. This quantization ensures that the agent can focus on relevant distances without being overwhelmed by excessively granular data.

## **Reward Shaping**

The reward function is critical in guiding the agent's learning process by encouraging desirable actions and penalizing undesirable ones.

First, we tried to give the agent a small positive reward for each step survives (+1), a big positive reward for successfully passing a pipe (+20), and a significant negative reward when the bird collides with the pipe or the ground (-20).

This method led the agent to pass an average of 2 pipes per game, so we decided to encourage the agent to keep the bird's vertical position as similar as possible to the next pipe center position. A penalty proportional to the bird's vertical misalignment with the center of the pipe was introduced (-2 \* \[player\_position – center\_position\]), this ensures that the agent focuses on aligning itself with the opening of the next pipe to minimize collision risks.

For these rewards shaping functions our score per game decreased, but still, the agent wasn’t stable, and the training process took a lot of time.

We decided to normalize the rewards so that the range of the rewards will be \[-1, 1\]:

* Survival reward: \+0.05  
* Penalty for misalignment: \-0.1 \* \[(player\_position – center\_position) / 512\]  
* Success reward: \+1  
* Failure reward: \-1

After several training sessions, we managed to get high scores, but the agent’s performance was inconsistent, and our success rate was around 60%. We decided to add a reward for getting closer to the center of a pipe because we want the agent to keep moving as much as he can to the center of the pipes, and we modified the alignment penalty:

* Survival reward: \+0.05  
* Reward for alignment: \+0.1  
* Penalty for misalignment: \-0.2 \* \[sqrt(player\_position – center\_position) / 512\]  
* Success reward: \+1  
* Failure reward: \-1

## **Agent implementation**

two reinforcement learning algorithms were implemented: Q-Learning and SARSA. Both algorithms aim to train the agent to successfully navigate the Flappy Bird environment by learning an optimal policy through iterative updates to the Q-table.

The Q-Learning is more aggressive in learning the optimal policy by assuming the agent always selects the best possible action in the next state, potentially leading to faster convergence but at the risk of overestimating Q-values.

The SARSA, being on-policy, considers the actual actions taken during exploration, resulting in a more cautious learning approach that might be more stable but slower to converge in some cases.

For both algorithms, the maximum number of steps per training episode was set to 2000 to ensure the agent had enough time to interact with the environment and that the learning process would be effective. Also, for both agents we used the same reward shaping function and the same preprocess function, leading to the same multi-dimensional Q-table size.

* Q-Learning implementation:  
  We used an exponential decay strategy for epsilon to balance exploration and exploitation over time.  
  The epsilon value update was ε=min+max-mine-decay∙episode.  
  This approach gradually reduces exploration as training progresses, allowing the agent to focus more on exploiting learned policies in later episodes.  
  The Q-table update was Qs,a←Qs,a+αr+γs',a' \-Qs,a.  
  This means the agent updates the Q-value based on the maximum estimated future reward for the next state, making it an off-policy algorithm. This enables the agent to optimize its policy independently of the action taken in the next state.

* SARSA implementation:  
  We used a linear decay strategy for epsilon was reduced stepwise but never fell below a minimum threshold.  
  The epsilon value update was ε=min, ε∙decay .  
  This approach ensures a more gradual decrease in exploration, potentially allowing the agent to continue exploring new states for a longer duration compared to the exponential decay method in Q-Learning.  
  The Q-table update was Qs,a←Qs,a+αr+γQs',a'-Qs,a.  
  Unlike Q-Learning, SARSA uses the Q-value of the specific action chosen in the next state, making it an on-policy algorithm. This approach ensures that the updates are more aligned with the agent's actual policy.

By implementing both algorithms, we could compare their performance under the same environment and training conditions, analyzing how the different epsilon update strategies and Q-value update rules impact the learning efficiency and overall success rate of the agent.

# Results

We used the same preprocess and reward shaping functions for both Q-Learning and SARSA agents to ensure consistency in the agent's learning process, while systematically exploring various hyperparameter configurations and conducting multiple experiments. 

As mentioned above, at the beginning of the training we used a reward shaping function that achieved a high reward and score rate, but the running average showed oscillations which indicated instability, as can be seen in Figure 1\.

To deal with it, we modify our reward shaping function, as described in the Methodology part.

### 

### 

### Q-Learning \- Training

we decided to focus and show those three experiments:

|  | Experiment 1 | Experiment 2 | Experiment 3 |
| :---- | :---- | :---- | :---- |
| **Gamma** | 0.5 | 0.7 | 0.9 |
| **Learning Rate** | 0.1 | 0.3 | 0.7 |
| **Epsilon Decay** | 0.0001 | 0.0005 | 0.001 |
| **Epsilon Start** | 1.0 | 1.0 | 1.0 |
| **Epsilon Min** | 0.01 | 0.01 | 0.01 |

Table 1 – Q-Learning Hyperparameters used in the experiments

For each experiment, we trained the agent for 10,000 episodes, and with the best of them, we will continue to a longer training.

In the first experiment, we chose a low gamma, low learning rate, and a very slow exploration decay. As can be seen in Figure 2, the training showed slow but steady improvement in rewards, but the score is almost zero for the entire training, indicating that the agent struggles to pass pipes. The reason for this is the small learning rate and the low gamma, which emphasizes short-term rewards, which hindered learning long-term strategies for survival.

In the second experiment, we chose a higher gamma, a moderate learning rate, and a moderate exploration decay. As can be seen in Figure 3, the training achieved higher rewards than experiment 1, a steady learning process and a high score of 50\. The balance hyperparameters helped the agent to learn efficiently.

For the third experiment, we chose the highest gamma, high learning rate, and a faster exploration decay. As can be seen in Figure 4, the training achieved the highest overall performance with the best reward progression and highest score (also in running average). However around 8000 episodes, we saw an interesting drop that can indicate some instability in the model.

From the experiments we can understand that:

* Higher gamma values led to better long-term reward optimization.  
* Faster epsilon decay worked well together with a higher learning rate, as can be seen in experiment 3\.  
* Higher learning rate showed batter results, maybe due to faster adaptation to new situations.

Although the oscillation appears in experiment 3, which can be due to exploration phase, we decided to continue with experiment 3 hyperparameters for longer training session.

Figure 5 shows the training of an agent with experiment 3 hyperparameters for 30,000 episodes. As can be seen, the agent successfully learns an optimal policy after 10,000 episodes and consistently achieves high rewards (around 150\) and scores (around 30).

After that, we decided to try and train the agent on the environment of a smaller pipe gap, to ensure the agent will be able to pass as many pipes as possible when the pipe gap equals 65\. Figure 6 shows the results of the training, which achieve a running average of score around 20, in contrast to 30 in the original training.

### 

### SARSA \- Training

we decided to focus and show those three experiments:

|  | Experiment 1 | Experiment 2 | Experiment 3 |
| :---- | :---- | :---- | :---- |
| **Gamma** | 0.5 | 0.8 | 0.95 |
| **Learning Rate** | 0.5 | 0.4 | 0.3 |
| **Epsilon Decay** | 0.995 | 0.999 | 0.9995 |
| **Epsilon Start** | 1.0 | 1.0 | 1.0 |
| **Epsilon Min** | 0.01 | 0.01 | 0.01 |

Table 2 – SARSA Hyperparameters used in the experiments

For each experiment, we trained the agent for 10,000 episodes, and with the best of them, we will continue to a longer training.

In the first experiment, we chose a moderate gamma, moderate learning rate, and a lower decay rate. As can be seen in Figure 7, the training showed improvement in both rewards and score, when the maximal score achieved 50, however, the agent struggled to converge effectively, with rewards and score oscillating significantly which may indicate about unstable process.

In the second experiment, we chose a higher gamma, lower learning rate, and a higher decay rate. As can be seen in Figure 8, the training achieved higher rewards than experiment 1, a steadier learning process, and a high score of 50\. The running average for rewards and scores showed noticeable improvement but still exhibited fluctuations, indicating suboptimal learning.

For the third experiment, we chose the highest gamma, the lowest learning rate, and the highest decay rate. As can be seen in Figure 9, the training achieved the best performance among the three experiments. The rewards increase steadily, with fewer oscillations, and the steps per episode and the score exhibit consistent improvement over episodes, indicating that the agent effectively balances exploration and exploitation.

From the experiments we can understand that:

* Higher gamma values led to better long-term strategic behavior.  
* A lower learning rate provides more stable learning but slower convergence.  
* Very high epsilon decay (0.9995) ensures a very gradual transition from exploration to exploitation.

Experiment 3 was the most stable and showed consistent improvement over episodes for the score, so we decided to continue with experiment 3 hyperparameters for longer training sessions.

Figure 10 shows the training of an agent with experiment 3 hyperparameters for 30,000 episodes. As can be seen, in the initial phase (0-5,000 episodes) the agent shows gradual improvement in all metrics, in the acceleration phase (5,000-10,000 episodes) the agent shows a rapid increase in performance, and in the stabilization phase (10,000-30,000 episodes) the agent maintains consistent performance.

After that, we decided to try and train the agent on the environment of a smaller pipe gap, to ensure the agent will be able to pass as many pipes as possible when the pipe gap equals 65\. Figure 11 shows the results of the training, which achieved a running average of score around 20, in contrast to 35 in the original training.

### Q-Learning – Validation

We validate our Q-Learning agent using two Q-tables, one table of an agent trained on a pipe gap of 80, and the second table of an agent trained on a pipe gap of 65\.

For each agent we validate, we run the environment for 10 episodes for each pipe gap (65 and 80), a total of 4 tests for each algorithm.

|  | Pipe Gap \= 80 | Pipe Gap \= 65 |
| :---- | :---: | :---: |
| **Success Rate** | 100% | 0% |
| **Mean Score** | 139.80 ± 97.97 | 1.80 ± 0.87 |
| **Best Score** | 351 | 3 |
| **Mean Episode Length** | 5325.8 | 105.9 |
| **Mean Total Reward** | 692.52 | 11.54 |

Table 3 – Q-Learning agent trained on pipe\_gap=80 validation summary

|  | Pipe Gap \= 80 | Pipe Gap \= 65 |
| :---- | :---: | :---: |
| **Success Rate** | 90% | 70% |
| **Mean Score** | 90.00 ± 93.17 | 36.10 ± 54.59 |
| **Best Score** | 293 | 197 |
| **Mean Episode Length** | 3449.5 | 1419.1 |
| **Mean Total Reward** | 432.83 | 177.63 |

Table 4 – Q-Learning agent trained on pipe\_gap=65 validation summary

As can be seen in Table 3 and Table 4, for a pipe\_gap=80 the agent performance decreased when he trained on a pipe\_gap=65 but was still able to maintain more than 90% success rate. However, for pipe\_gap=65, there was a large improvement when we trained the agent on pipe\_gap=65, and he was able to get a 70% success rate instead of 0%.

Figure 12-15 shows a graphical summary of the agent’s performance, including the score, steps, and rewards per episode and a boxplot of the score distribution.

### SARSA – Validation

We validate our SARSA agent using two Q-tables, one table of an agent trained on a pipe gap of 80, and the second table of an agent trained on a pipe gap of 65\.

For each agent we validate, we run the environment for 10 episodes for each pipe gap (65 and 80), a total of 4 tests for each algorithm.

|  | Pipe Gap \= 80 | Pipe Gap \= 65 |
| :---- | :---: | :---: |
| **Success Rate** | 100% | 70% |
| **Mean Score** | 93.60 ± 80.89 | 15.50 ± 11.88 |
| **Best Score** | 286 | 47 |
| **Mean Episode Length** | 3585.6 | 630.0 |
| **Mean Total Reward** | 439.91 | 74.66 |

Table 5 – SARSA agent trained on pipe\_gap=80 validation summary

|  | Pipe Gap \= 80 | Pipe Gap \= 65 |
| :---- | :---: | :---: |
| **Success Rate** | 100% | 100% |
| **Mean Score** | 84.90 ± 67.04 | 65.50 ± 31.78 |
| **Best Score** | 257 | 119 |
| **Mean Episode Length** | 3257.4 | 2526.5 |
| **Mean Total Reward** | 389.41 | 301.94 |

Table 6 – SARSA agent trained on pipe\_gap=65 validation summary

As can be seen in Table 5 and Table 6, for a pipe\_gap=80 the agent performance remains 100% success rate when trained on both pipe\_gap but for pipe\_gap=80 the performance was better. However, for pipe\_gap=65, there was an improvement when we trained the agent on pipe\_gap=65, and he was able to get a 100% success rate instead of 70%.

Figure 16-19 shows a graphical summary of the agent’s performance, including the score, steps, and rewards per episode and a boxplot of the score distribution.

# Discussion

As can be seen in the results section, both algorithms managed to get a high success rate in the flappy bird environment.

The Q-Learning agent performed better in a familiar environment (like the training, for example, training on pipe\_gap=80 and tested on pipe\_gap=80) but achieved lower results in an unfamiliar environment (different from the training, for example, trained on pipe\_gap=80 and test on pipe\_gap=65). On the other hand, the SARSA agent while slightly less effective compared to the Q-Learning in the familiar environment, showed better adaptability in the challenging (pipe\_gap=65) scenario. This aligns with the SARSA algorithm's tendency to prioritize safer actions due to its on-policy nature.

Also, the SARSA’s score distributions were more consistent, when the Q-Learning exhibited a wider variance in performance, which may be attributed to the SARSA’s action-selection strategy that balances exploration and exploitation.

The Q-Learning agent focused on maximizing long-term rewards, often leading to higher peak scores but lower adaptability, while the SARSA agent, prioritized immediate actions based on the current policy, resulting in more cautious but adaptable performance.

This project provided us with an opportunity to implement a reinforcement learning agent to solve a difficult environment.

We encountered several challenges during the project:

* Reward shaping – while finding the reward shaping function that will improve our agent performance the most, and trying several ways, we understood the importance of a good and balanced rewards system, and how it can affect the agent performance.

* Hyperparameters tuning – after training the agents with multiple parameters, and trying many variances and combinations, we realized that small changes in the hyperparameters could significantly influence the learning process and agent performance.

* Adaptability of agents – observing the agents' performance across different pipe gaps revealed the limitations of RL algorithms in generalization.

### Conclusions:

In conclusion, in this project we’ve managed to implement and compare Q-Learning and SARSA algorithms to train agents capable of navigating the Flappy Bird environment.   
The results demonstrated that both agents can achieve high performance, with Q-Learning generally excelling in more stable environments and SARSA showing better adaptability to dynamic settings. However, the performance was sensitive to hyperparameter choices, emphasizing the importance of effective tuning and reward shaping.

For future improvement, we can focus on on designing a more sophisticated reward-shaping mechanism to better fit the agent's behavior with long-term objectives for example.   
Further tuning hyperparameters through systematic trial and error exploration or with optimization techniques can enhance stability and learning efficiency.   
Identifying the optimal pipe gap during training that maximizes the agent's success rate across various gaps could significantly improve the robustness and adaptability of the agent in diverse environments. These improvements would contribute to more effective and generalizable reinforcement learning solutions.

## 

## Appendix

##### Figures Graphics and Metrics  

![][image3]

**Figure 1 – Example for experiment with old reward shaping function**

![][image4]

**Figure 2 – Q-Learning experiment no.1 results**

![A graph with blue linesDescription automatically generated][image5]

**Figure 3 – Q-Learning Experiment No.2 results**

![][image6]

**Figure 4 – Q-Learning Experiment No.3 results**

![A blue graph with white textDescription automatically generated][image7]

**Figure 5 – Q-Learning chosen hyperparameter training results**

![][image8]

**Figure 6 – Q-Learning chosen hyperparameter training results, pipe\_gap \= 65**

![][image9]

**Figure 7 – SARSA experiment no.1 results**

![][image10]

**Figure 8 – SARSA experiment no.2 results**

![A graph with a blue lineDescription automatically generated][image11]

**Figure 9 – SARSA experiment no.3 results**

![][image12]

**Figure 10 – SARSA chosen hyperparameter training results**

![][image13]

**Figure 11 – SARSA chosen hyperparameter training results, pipe\_gap \= 65**

![][image14]

**Figure 12 – Q-Learning agent trained on pipe gap 80 performance, pipe\_gap \= 80**

![A group of graphs with blue linesDescription automatically generated][image15]

**Figure 13 – Q-Learning agent trained on pipe gap 80 performance, pipe\_gap \= 65**

![A group of graphs with blue linesDescription automatically generated][image16]

**Figure 14 – Q-Learning agent trained on pipe gap 65 performance, pipe\_gap \= 80**

![A group of graphs with blue linesDescription automatically generated][image17]

**Figure 15 – Q-Learning agent trained on pipe gap 65 performance, pipe\_gap \= 65**

![A group of graphs with blue linesDescription automatically generated][image18]

**Figure 16 – SARSA agent trained on pipe gap 80 performance, pipe\_gap \= 80**

![A group of graphs with linesDescription automatically generated with medium confidence][image19]

**Figure 17 – SARSA agent trained on pipe gap 80 performance, pipe\_gap \= 65**

![][image20]

**Figure 18 – SARSA agent trained on pipe gap 65 performance, pipe\_gap \= 80**

![A group of graphs with numbersDescription automatically generated with medium confidence][image21]

**Figure 19 – SARSA agent trained on pipe gap 65 performance, pipe\_gap \= 65**

[image1]: figures/logo_small.png

[image2]: figures/flappybird_logo.png

[image3]: figures/old_reward_shaping_example.png

[image4]: figures/qlearning_training_agent1.png

[image5]: figures/qlearning_training_agent2.png

[image6]: figures/qlearning_training_agent3.png

[image7]: figures/qlearning_training_agent3_extended_gap80.png

[image8]: figures/qlearning_training_agent3_extended_gap65.png

[image9]: figures/sarsa_training_agent1.png

[image10]: figures/sarsa_training_agent2.png

[image11]: figures/sarsa_training_agent3.png

[image12]: figures/sarsa_training_agent3_extended_gap80.png

[image13]: figures/sarsa_training_agent3_extended_gap65.png

[image14]: figures/qlearning_eval_gap80_rewards.png

[image15]: figures/qlearning_eval_gap80_scores.png

[image16]: figures/qlearning_eval_gap65_rewards.png

[image17]: figures/qlearning_eval_gap65_scores.png

[image18]: figures/sarsa_eval_gap80_rewards.png

[image19]: figures/sarsa_eval_gap80_scores.png

[image20]: figures/sarsa_eval_gap65_rewards.png

[image21]: figures/sarsa_eval_gap65_scores.png
