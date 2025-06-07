# ff-draft-optimizer

## Executive Summary

This project explores the use of Reinforcement Learning (RL) to optimize drafting strategies in season-long NFL Fantasy Football leagues. The core question is: Given the current draft state—including roster needs, opponents' strategies, and remaining player pool—what is the optimal pick to maximize season-long performance? Leveraging public data from [NFL FastR](https://www.nflfastr.com/) and [FantasyPros](https://www.fantasypros.com/nfl/adp/ppr-overall.php), the project simulates thousands of drafts to train an RL agent (or multiple RL agents) that learns to make high-leverage decisions. The minimum viable product is capable of recommending optimal draft picks, balancing positional scarcity and long-term value in a competitive environment. Key challenges include effective simulation of draft dynamics and model training. This work builds upon prior methods such as Monte Carlo Tree Search and predictive modeling, aiming to offer a more adaptable, strategic solution through RL.

## Motivation

As a fantasy sports enthusiast, I often wonder how I can best optimize my drafting strategy. Given the dynamic environment, positional scarcity, and adversarial behavior from fellow drafters, putting your team in an optimal position to win the season can prove difficult. However, what if I could train a model that suggests the best possible pick given the current roster needs, opponent needs, and available player pool? Could this model ‘play’ itself thousands of times to find the best strategy in the draft?

## Data Question

Given the current draft state (i.e. team’s positional needs, opponents needs, player pool, etc.), what is the optimal pick in an NFL Fantasy Football draft?


Some people have attempted this by training models to predict fantasy football performance on a weekly basis using rolling player stats, team, opponent, and home vs. away game features. Predicting a player’s performance could provide a list of overall best players, but does not take into consideration the dynamic landscape of a fantasy draft, where drafting three top quarterbacks is highly unlikely to produce the best fantasy team. In simulating the draft, others have used Monte Carlo Tree Search to estimate the best current choice. Meanwhile, others have used dynamic programming to pick which position is most likely to add value to the team, thereby outperforming the AutoDraft. Some have opted for Deep Reinforcement Learning to select the best available player in a position, with moderate success. Carlos Fonseca noted that his RL model began making sensible decisions and improved his team’s score. While some have attempted to use Reinforcement Learning, I have not found any that attempt any kind of multi-agent RL that further optimizes for the dynamic strategies made by other drafters.


## Minimum Viable Product

The minimum viable product is a model that will suggest which player I should draft given the current draft board, thereby optimizing my team’s expected season total points. This doesn’t need to be implemented with an actual draft website (like Yahoo, ESPN, or Sleeper), although it would be nice for practicality.

## Schedule

Get the Data (5/20/2025)
Clean & Explore the Data (5/24/2025)
Create Presentation (6/21/2025)
Internal Demos (6/28/2025)
Graduation (7/3/2025)
Demo Day (7/10/2025)


## Data Sources

- **NFL FastR**: [https://www.nflfastr.com/](https://www.nflfastr.com/)
- **FantasyPros ADP Data**: [https://www.fantasypros.com/nfl/adp/ppr-overall.php](https://www.fantasypros.com/nfl/adp/ppr-overall.php)

## Known Issues and Challenges

Training is likely a problem. I’m prepared to let my computer run for long periods of time. I also believe Kaggle may provide some boosts in training speed. Additionally, finding ways to work within the constraints of my GPU is part of the fun of the project… How many features can I add until this gets too out of hand? It seems from my research that too many features frequently make models worse, so we will see what I can do to optimize the model.
