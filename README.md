# ff-draft-optimizer

## Executive Summary

This project explores the use of Reinforcement Learning (RL) to optimize drafting strategies in season-long NFL Fantasy Football leagues. The core question is: Given the current draft state—including roster needs, opponents' strategies, and remaining player pool—what is the optimal pick to maximize season-long performance? Leveraging public data from [NFL FastR](https://www.nflfastr.com/) and [FantasyPros](https://www.fantasypros.com/nfl/adp/ppr-overall.php), the project simulates thousands of drafts to train multiple RL agents to make high-leverage decisions. The minimum viable product is capable of recommending optimal draft picks, balancing positional scarcity and long-term value in a competitive environment. Key challenges include effective simulation of draft dynamics and model optimization. This work builds upon prior methods such as Monte Carlo Tree Search and predictive modeling, aiming to offer a more adaptable, strategic solution through RL.

## Introduction to Fantasy Football

Fantasy football is a game in which several managers take turns drafting real football players onto their fantasy teams before the NFL season begins, and go on to compete head-to-head each week to see who’s fantasy teams perform best based on a predetermined scoring system. This scoring system uses player statistics to calculate fantasy scores. For example, catching a touchdown may net your player 6 fantasy points, while a fumble will cost them 2. A typical fantasy team may have something like 1 QB, 2 RBs, 2 WRs, 1 TE, 1 flex player, 1 defense, and 1 kicker along with a bench.

## Motivation

Fantasy football success begins at the draft. While it is possible to pick up free agents and trade players with other managers in your fantasy league, the primary method of season-long success is skillfully navigating the highly dynamic draft. Picking the players that will go on to succeed in the season can be complicated by factors such as injuries, schedules, and NFL team strategies, leading managers to often overlook some of the highest-value players of the season. Of course, these high-value players cannot always be detected. This is where in-draft strategy also becomes key. How does one balance the risk-reward in drafting a player now, or waiting and potentially losing them to another manager? How does one navigate positional value with more reliable but less positionally-scarce players?

Thus, this project was created to find the optimal drafting strategy in any given draft state. To do this, we have made several simplifying [assumptions]( https://github.com/alexanderjireland/ff-draft-optimizer?tab=readme-ov-file#assumptions-made). To predict player success and variability, we used [Bayesian Linear Regression](https://brunaw.com/phd/bayes-regression/report.pdf) to create Posterior Predictive Distributions of individual players. These predictions were then fed into a [Multi-Agent Reinforcement Learning](https://github.com/LantaoYu/MARL-Papers) model using [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) to develop drafting strategies.

## Data Question

Given the current draft state (i.e. team’s positional needs, opponents needs, player pool, etc.), what is the optimal pick in an NFL Fantasy Football draft?

Some people have attempted this by training models to predict fantasy football performance on a weekly basis using rolling player stats, team, opponent, and home vs. away game features. Predicting a player’s performance could provide a list of overall best players, but does not take into consideration the dynamic landscape of a fantasy draft, where drafting three top quarterbacks is highly unlikely to produce the best fantasy team. In simulating the draft, others have used Monte Carlo Tree Search to estimate the best current choice. Meanwhile, others have used dynamic programming to pick which position is most likely to add value to the team, thereby outperforming the AutoDraft. Some have opted for Deep Reinforcement Learning to select the best available player in a position, with moderate success. Carlos Fonseca noted that his RL model began making sensible decisions and improved his team’s score. While some have attempted to use Reinforcement Learning, I have not found any that attempt any kind of multi-agent RL that further optimizes for the dynamic strategies made by other drafters.


## Minimum Viable Product

The minimum viable product is a model that will suggest which player I should draft given the current draft board, thereby optimizing my team’s expected season total points. This doesn’t need to be implemented with an actual draft website (like Yahoo, ESPN, or Sleeper), although it would be nice for practicality.

## Schedule

Get the Data (5/20/2025) \
Clean & Explore the Data (5/24/2025) \
Create Presentation (6/21/2025) \
Internal Demos (6/28/2025) \
Graduation (7/3/2025) \
Demo Day (7/10/2025)

## Data Sources

- **NFL FastR**: [https://www.nflfastr.com/](https://www.nflfastr.com/)
- **FantasyPros ADP Data**: [https://www.fantasypros.com/nfl/adp/ppr-overall.php](https://www.fantasypros.com/nfl/adp/ppr-overall.php)

## Assumptions Made

We have made several critical assumptions in this project. First, Kickers and Defenses have been intentionally excluded from consideration, as their performance is frequently tied to weekly NFL matchups, encouraging a "streaming" approach (i.e. managers swap their players with free agents expected to perform better that week) that offers minimal strategic depth in the draft phase. The team composition is based on a classic lineup: one Quarterback, two Running Backs, two Wide Receivers, one Tight End, and one FLEX player (eligible for RB, WR, or TE), complemented by seven Bench players. Player fantasy performance calculations were derived from weeks 1 through 17 of each NFL season since 1999. This was done to align with the typical end date of most fantasy leagues. Additionally, we define "significant injury time" as any absence lasting a minimum of four weeks, making a player "injury prone" if they've endured such an absence twice or more within the preceding three seasons. All player scoring is calculated using [ESPN's standard PPR (Points Per Reception) format](https://support.espn.com/hc/en-us/articles/360003914032-Scoring-Formats). To simplify the draft environment while still encouraging positional strategy, our Reinforcement Learning Draft Environment currently presents only the highest-projected available player at each position. Finally, the model currently only incorporates two agents playing each other in a linear draft order.

## Why Bayesian Linear Regression?

 Bayesian Linear Regression is helpful in dealing with insufficient data while also allowing one to ask how confident we are in the fitted data. This allows us to create a distribution of probable outcomes for specific players. We can then answer questions probabilistically, by saying, for example, that we are 95% confident a player will score between 200 and 250 fantasy points this season. We can also calculate the probability that a player scores over a certain number of points. Feeding this information into the Reinforcement Learning algorithm allows the model to take into consideration uncertainty in player outcomes while also gaining insight into their overall fantasy value.

## Why Multi-Agent Reinforcement Learning?

Reinforcement Learning is a powerful tool for learning optimal strategies in highly complex environments. Applied to fantasy drafts, we can reward the model for “successful” behavior while penalizing unsuccessful behavior. Furthermore, by incorporating multiple agents in the same environment, we are able to replicate a real fantasy draft scenario in which each manager is incentivized to maximize their own likelihood of success while minimizing others’. Our model is rewarded in two ways: first, it is rewarded at each draft selection based upon that pick’s Value Over Replacement (how much better is this pick projected to be than the next best at that position) and Hurt Score (how many points will the next drafter potentially lose, given that this is a positional need for them). At the end of the draft, teams’ true fantasy scores (based upon the 2024 season) are calculated, and the winning team is decided by the highest scoring optimal lineup of starters. This team is rewarded, while all other teams are penalized based on the difference between their score and the winning team’s score.

## Libraries Used

[pymc](https://www.pymc.io/welcome.html) — for Bayesian regression modeling

[arviz](https://python.arviz.org/en/stable/) — for visualizing and analyzing Bayesian models

[ray[rllib]](https://docs.ray.io/en/latest/rllib/index.html) — for training reinforcement learning agents

[gymnasium](https://gymnasium.farama.org/) — for creating custom multi-agent draft environments

[pettingzoo](https://pettingzoo.farama.org/) - also for creating custom multi-agent draft environments

[scikit-learn](https://scikit-learn.org/stable/) - for imputing data, preprocessing, and developing basic regression models

## Future Work

There is a lot that can still be done, including optimizing both the Bayesian Regression and RL models. There is still more data that can be fed into both these models and there is room for hyperparameter optimization for the models as well. Additionally, the drafting model has only seen two agents playing each other. Increasing this number, while also finding more optimal ways to reward the agents, may lead to more complex and forward-thinking strategies. Expanding the model’s action space to include a wider selection of players, rather than the top player in each position, may also lend itself to deeper strategic development. Finally, a way to implement the draft pick suggestions right into an actual fantasy draft environment would be ideal for practical use.

## Known Issues and Challenges

Optimally rewarding the RL model can be difficult. The algorithm may develop a policy that gets "stuck" in local optima, rather than the universal optimum.