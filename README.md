# Stock Agent Analysis Application

An interactive stock research, trading agent, and backtesting dashboard powered by Streamlit, yfinance, and NLTK VADER sentiment analysis.

---

## Features
- **Stock Analysis**: View historical stock charts, moving averages, and technical indicators (RSI, MACD, Bollinger Bands).
- **Sentiment Analysis**: Analyze recent headlines and industry news to calculate an average sentiment score.
- **Agent Action Signals**: Compute technical & sentiment-based BUY, SELL, or HOLD recommendations.
- **Backtesting Dashboard**: Simulate performance metrics (Total Return, Sharpe Ratio, Max Drawdown) against a standard Buy-and-Hold strategy.

---

## 🚀 Options to Share the Streamlit UI

There are three ways you can share this app with your friend:

### Option A: Sharing on the Same Local Network (Wi-Fi)
If your friend is connected to the **same Wi-Fi/local network**, they can access your running Streamlit server directly.
1. Run the app on your machine:
   ```bash
   python -m streamlit run app.py
   ```
2. Note the **Network URL** shown in the terminal (usually `http://<your-local-ip>:8501`). For example:
   ```
   http://192.168.1.16:8501
   ```
3. Send this URL to your friend. Make sure your local firewall allows inbound connections on port `8501`.

### Option B: Sharing Globally via Ngrok (No setup on friend's system)
If your friend is not on the same Wi-Fi network, you can create a secure public tunnel to your local machine using **ngrok**.
1. Download and install [ngrok](https://ngrok.com/).
2. Run your Streamlit app locally:
   ```bash
   python -m streamlit run app.py
   ```
3. Expose the port in a separate terminal:
   ```bash
   ngrok http 8501
   ```
4. Copy the generated public forwarding URL (e.g. `https://xxxx-xx-xx.ngrok-free.app`) and send it to your friend. They can open it anywhere in their browser!

### Option C: Share the Code (Running Locally on Their System)
You can zip this project folder or upload it to GitHub, and your friend can run it on their system by following the installation guide below.

---

## 🛠️ Installation & Setup (For running locally)

### Prerequisites
Make sure you have **Python 3.8+** installed.

### 1. Clone or Download the Project
Extract the project folder onto your machine.

### 2. Install Dependencies
Navigate to the directory and run:
```bash
pip install -r requirements.txt
```

### 3. Run the App
Start the Streamlit server:
```bash
python -m streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501`.
