# LLM-Enhanced Reinforcement Learning Trading Agent

## Project Goal
Train a **reinforcement learning (RL)** agent that learns how to trade stocks based on historical market data from Yahoo Finance.
The agent makes autonomous **Buy/Sell/Hold decisons** to maximize cumulative returns.
Uses a lightweight **LLM/NLP** pipeline which converts news or social posts into numeric sentiment score, which is then used as an additional feature for the trading agent.

## Features
1. Custom trading environment simulating realistic market dynamics.
2. Reinforcement learning agent using **`Q-Learning`**.
3. Technical indicators (SMA, EMA, RSI) as features for better decision-making.
4. Uses **VADER (NLTK)** for very fast, dependency-light sentiment scores 
5. Jupyter notebooks for data exploration and agent training visualization. 
6. Unit tests ensuring enviroment consistency and agent correctness.

## Tech Used
- **Python** (pandas, numpy, matplotlib)
- **NLTK VADER** for sentiment
- **Reinforcement Learning (Q-Learning)**
- **Jupyter Notebook** for experimentation

## Future Plans
- Replace Q-Learning with a more advanced reinforment learning algorthim (i.e., D`eep Q Netowrk`, `PPO`, etc)
- Add real `LLM nes sentiment` via OpenAI or Hugging Face API
- Introduce `multi-stock portfolio management`
- Deploy a `dashboard with Streamlit` for live trading simulation
