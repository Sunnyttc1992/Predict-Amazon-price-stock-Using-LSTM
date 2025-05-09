### Hello and welcome!
My name is Sunny, and this repository showcases a deep learning project focused on predicting Amazon stock prices using an LSTM (Long Short-Term Memory) neural network.

## 🧠 Project Overview

This project demonstrates how LSTM models can be applied to time series forecasting in the financial domain. Specifically, the model is trained to predict the closing price of Amazon (AMZN) stock based on historical data.

## 🔍 Scenario

The historical stock data for Amazon was sourced from Kaggle.

A sliding time window of 60 days is used as input to predict the stock price on the 61st day.

The LSTM model captures temporal patterns in the stock market data and attempts to forecast future prices based on trends in the past.

## ⚙️ Techniques & Tools

Python, Pandas, and NumPy for data preprocessing.

Matplotlib for visualization.

TensorFlow/Keras for building and training the LSTM model.

## 📁 Dataset

Dataset used: Amazon_Stock_Price.csv (from Kaggle)

Features include Open, High, Low, Close, and Volume.

## 📊 Outcome
The trained LSTM model is evaluated on unseen data, and its predictions are plotted against actual stock prices to visualize performance.
