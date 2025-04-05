# 📊 A/B Testing Simulator

An interactive Streamlit application that simulates and analyzes A/B tests using advanced experimentation techniques.

---

## 🚀 Live App
Check out the deployed app here:
👉 [https://A-B-Test-Simulator.streamlit.app](https://A-B-Test-Simulator.streamlit.app) *(replace with your actual link after deploying)*

---

## 🧪 Features
- **Synthetic Data Simulation**: Generate realistic user-level A/B test data with configurable sample size, treatment effects, dropouts, and stratification.
- **CUPED Adjustment**: Reduces variance using pre-experiment metrics.
- **Classical A/B Testing**: Two-sample t-tests with significance level.
- **Bayesian A/B Testing**: Posterior inference and probability treatment > control.
- **Uplift Modeling**: Heterogeneous treatment effect estimation using ML.
- **Interactive Visualizations**: Metric distributions and CUPED-adjusted plots.

---

## 🗂️ Project Structure
```
A-B-Test-Simulator/
├── dashboard/
│   └── app.py                # Streamlit dashboard UI
├── src/
│   ├── simulate_experiment.py
│   ├── cuped.py
│   ├── inference_methods.py
│   ├── bayesian_ab.py
│   ├── uplift_model.py
│   └── evaluation.py
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 💻 Run Locally
### 1. Clone the repository:
```bash
git clone https://github.com/IshitaLohia/A-B-Test-Simulator.git
cd A-B-Test-Simulator
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app:
```bash
streamlit run dashboard/app.py
```

---

## ⚙️ How It Works

This simulator allows you to design and evaluate A/B tests by controlling key experimental parameters and visualizing the outcomes. Here's how it works:

1. **Simulate Data**: You define the number of users, expected treatment effect, dropout rate, and whether to stratify by a feature (like device type).

2. **Apply CUPED**: A variance reduction technique that adjusts post-treatment metrics using pre-treatment data to increase statistical power.

3. **Analyze Results**:
   - **Classical Inference**: Uses a t-test to compute the p-value and confidence intervals.
   - **Bayesian Inference**: Estimates the posterior probability that treatment outperforms control.
   - **Uplift Modeling**: Identifies segments of users who are more or less likely to benefit from the treatment.

4. **Visualize Outputs**: Real-time plots of pre/post distributions and CUPED-adjusted metrics help users interpret effects.

---

## 🧠 Future Enhancements
- Add sequential testing and early stopping rules
- Integrate power calculation module
- Upload and analyze real-world test logs (CSV input)

---

## 👨‍💻 Author
*Built with ❤️ by Ishita Lohia*
- [GitHub Profile](https://github.com/IshitaLohia)
- [LinkedIn](https://linkedin.com/in/yourprofile)

