# A/B Testing Simulator

An interactive Streamlit application that simulates and analyzes A/B tests using advanced experimentation techniques.

---

## Live App
Check out the deployed app here:
[https://a-b-test-simulator-run.streamlit.app/](https://a-b-test-simulator-sim.streamlit.app/)

---

##  Features
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

## 🖥️ Interactive Dashboard

This project includes a powerful Streamlit dashboard that allows users to run A/B test simulations interactively without writing code.

###  What You Can Control
Use the sidebar sliders and inputs to modify:
- **Sample Size** – Adjust total number of users
- **Treatment Effect (%)** – Simulate different uplift strengths
- **Dropout Rate** – Control missing post-experiment data
- **Stratification** – Toggle stratified randomization
- **Random Seed** – Generate reproducible or new samples

###  What You Can See
The dashboard displays:
- **Results Summary** (p-value, posterior probability, effect size)
- **Practical Significance** based on delta thresholds
- **Distribution plots** of:
  - Raw post-treatment metric
  - CUPED-adjusted metric

###  Bonus Features
- CUPED variance reduction comparison
- Uplift modeling results (average uplift, AUC)

---

## Future Enhancements
- Add sequential testing and early stopping rules
- Integrate power calculation module
- Upload and analyze real-world test logs (CSV input)

---

## 👨‍💻 Author
Built with ❤️ by Ishita Lohia  
[GitHub Profile](https://github.com/IshitaLohia)  
[Linkedin Profile](https://www.linkedin.com/in/ishita-lohia-469551122/)  


