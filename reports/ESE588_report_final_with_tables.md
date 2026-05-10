Adaptive Ensemble Stock Forecasting with Context-Aware Stacking

Swetha Lekkalapudi
ESE 588
May 10, 2026

1. Introduction

Financial forecasting is a difficult machine learning problem because market data are noisy, nonstationary, and sensitive to regime changes. Even when predictive structure exists, it is often weak and unstable over time. For this reason, it is unlikely that any single model class will perform best across all market conditions. This project studies a short-horizon prediction task: determining whether the price of a security will be higher after five trading days.

The real-data study focuses on SPY, the SPDR S&P 500 ETF Trust. SPY was selected as the initial benchmark because it is smoother than many individual equities and therefore provides a cleaner environment for testing a forecasting system. The five-day horizon was chosen because one-day prediction is often dominated by noise, while much longer horizons mix short-term technical patterns with broader macroeconomic drift.

The main motivation of the project is to study whether combining heterogeneous models can improve forecasting. Three base learners were selected to represent different inductive biases. ARIMAX models structured linear dependence with exogenous inputs. XGBoost models nonlinear interactions in engineered indicators. LSTM models sequential dependence across rolling temporal windows. These models were then combined using both logistic stacking and context-aware stacking.

The project does not assume in advance that the ensemble must win. Instead, it asks a more careful question: when does ensemble learning help, and when does adaptive stacking fail to generalize? To answer that question, the project includes both synthetic and real-data experiments. The synthetic experiments are used to test model behavior under controlled linear, nonlinear, sequential, and regime-switching settings. The SPY experiment is then used to evaluate whether those same conclusions carry over to real historical market data.

2. Problem Formulation

Let P_t denote the adjusted closing price at trading day t. The primary target is a binary directional label:

y_t = 1 if P_(t+5) > P_t, else 0

This label indicates whether the asset price is higher after five trading days.

The project also defines a continuous forward return:

r_t_5 = (P_(t+5) / P_t) - 1

This secondary target is used by the ARIMAX model, which naturally produces return forecasts rather than class probabilities.

The supervised learning problem is to learn a mapping

f(x_t) -> p_hat_t

where x_t is the feature vector available at time t and p_hat_t is the estimated probability that y_t = 1. The final prediction rule is:

y_hat_t = 1 if p_hat_t >= threshold, else 0

For most models the default threshold is 0.5, although the final context-aware stacker uses a validation-selected threshold.

The engineered features include:
MA5, MA20, MA50
trend_signal = MA5 - MA20
moving average crossover indicator
vol5, vol10, vol20
RSI
MACD
momentum5
daily return
lag_1 through lag_5
volume_ma20
volume_change

The main assumptions are:
1. Historical price and volume information contain weak but useful predictive structure.
2. The relationship between features and future direction is regime-dependent.
3. Different model classes capture different forms of structure.
4. Chronological splitting is necessary to avoid temporal leakage.

The real-data experiment uses the following chronological split:
Train: 2013-01-01 to 2021-12-31
Validation: 2022-01-01 to 2022-12-31
Test: 2023-01-01 to 2024-01-01

This ensures that base-model training, meta-model training, and final evaluation are separated in time.

3. Method

The project implements three base learners and two stacking models.

3.1 ARIMAX

The first base learner is ARIMAX, implemented using SARIMAX from statsmodels. It predicts the 5-day forward return as a function of past return structure and exogenous signals:

r_t_5 = g(past_returns, exogenous_features) + error_t

The exogenous variables are:
trend_signal
vol10
rsi

Three candidate orders were evaluated:
(1, 1, 1)
(2, 1, 2)
(3, 1, 1)

The final model was selected using validation accuracy, with AIC used as a secondary criterion. The directional output is obtained by thresholding the predicted return:

y_hat_ARIMAX_t = 1 if r_hat_t_5 > 0, else 0

ARIMAX serves as an interpretable econometric baseline.

3.2 XGBoost

The second base learner is XGBoost, implemented using XGBClassifier. The model is trained on the full engineered feature set with 250 trees, maximum depth 4, learning rate 0.05, and row and column subsampling. XGBoost is intended to capture nonlinear interactions among technical indicators, lagged returns, and volume-based features without imposing a linear structure.

3.3 LSTM

The third base learner is an LSTM implemented in TensorFlow/Keras. The model uses 20-day rolling windows with the following sequential inputs:
trend_signal
vol10
rsi
lag_1 to lag_5

The architecture is:
Input
LSTM with 32 hidden units
Dropout with rate 0.2
Dense sigmoid output

Features are scaled using MinMaxScaler fitted only on the training set. The LSTM is designed to capture temporal dependence that static tabular models may miss.

3.4 Stacking and Context-Aware Stacking

Two stackers are implemented.

The first is logistic regression stacking using only base model outputs:
xgb_prob
lstm_prob
arimax_prob

The second is context-aware XGBoost stacking. In the final version of the project, the meta-model uses:
xgb_prob
lstm_prob
arimax_prob
vol10
rsi
trend_signal

This reduced context set was chosen after earlier versions with larger context spaces overfit strongly. The guiding idea is that model reliability may depend on market regime, but the meta-learning stage should focus only on the most stable regime indicators instead of trying to exploit every available technical feature.

The final context-aware stacker also includes several anti-overfitting modifications:
shallower trees
fewer boosting rounds
stronger L1 and L2 regularization
reduced row and column subsampling
chronological holdout within the validation period
early stopping
validation-based threshold selection

These changes were introduced because the initial adaptive stacker achieved unrealistically high validation performance and generalized poorly to the test period.

3.5 Evaluation

The project uses both classification and financial evaluation.

Classification metrics:
accuracy
precision
recall
F1 score
confusion matrix

Financial metrics:
cumulative return
Sharpe ratio

The trading rule is long/flat:
long if prediction = 1
flat if prediction = 0

Table 1. Summary of model classes, input types, output targets, intended forecasting roles, and their main strengths and limitations.

Model: ARIMAX
Input: Forward-return series with exogenous variables (trend_signal, vol10, rsi)
Output: 5-day return forecast converted to direction
Main purpose: Linear econometric baseline
Strength: Interpretable and suited for smooth linear structure
Limitation: Struggles with nonlinear interactions and complex classification boundaries

Model: XGBoost
Input: Full engineered feature set
Output: Probability of price increase after 5 days
Main purpose: Nonlinear tabular classifier
Strength: Captures nonlinear feature interactions well
Limitation: Does not explicitly model temporal sequence structure

Model: LSTM
Input: 20-day rolling windows of sequential indicators
Output: Probability of price increase after 5 days
Main purpose: Sequence-based temporal model
Strength: Captures temporal dependence and persistent hidden-state behavior
Limitation: More data-hungry and less interpretable

Model: Logistic Stacking
Input: Base model probabilities
Output: Final ensemble probability
Main purpose: Static ensemble baseline
Strength: Simple and often more stable than complex meta-models
Limitation: Cannot adapt weights based on regime or context

Model: Context-Aware XGBoost Stacking
Input: Base model probabilities plus stable context features (vol10, rsi, trend_signal)
Output: Final ensemble probability
Main purpose: Adaptive regime-aware ensemble
Strength: Can learn conditional model reliability with a compact context space
Limitation: Still vulnerable to unstable meta-learning and limited validation data

4. Experiments on Synthetic Data

Synthetic experiments were designed to test whether model performance aligns with the structure of the data-generating process. Synthetic returns were generated first, then converted to price paths using:

P_t = P_(t-1) * (1 + r_t)

Synthetic volume was also generated so that the real feature-engineering pipeline could be reused. Each synthetic dataset was split chronologically into training, validation, and test partitions.

Four synthetic regimes were studied:
1. linear autoregressive regime
2. nonlinear threshold regime
3. sequential latent-state regime
4. regime-switching regime

Noise levels were varied across low, medium, and high settings. A sample-size sensitivity study was also performed for the regime-switching case.

4.1 Linear Regime

The linear regime followed:

r_t = 0.25 * r_(t-1) + 0.015 * z_t + error_t

This setting was intended to favor ARIMAX. However, ARIMAX did not dominate the final 5-day classification task. At medium noise, ARIMAX achieved test F1 of 0.1433, while logistic stacking and context-aware stacking achieved test F1 values of 0.6746 and 0.6617, respectively. This indicates that even when the underlying return process is linear, the transformed classification problem based on multi-day labels and engineered features can be more complex than the original return equation.

4.2 Nonlinear Threshold Regime

The nonlinear regime introduced threshold-dependent behavior and was intended to favor XGBoost. XGBoost performed competitively, especially at low noise, but logistic stacking was often the strongest model overall. At medium noise, logistic stacking achieved test F1 of 0.6720, outperforming both XGBoost and context-aware stacking. These results suggest that nonlinear structure benefits from model combination, not only from a single nonlinear learner.

4.3 Sequential Latent-State Regime

The sequential regime introduced persistent hidden states and temporal memory. This was the clearest success case for the LSTM. At low noise, LSTM achieved test accuracy of 0.6592 and test F1 of 0.7319, clearly outperforming ARIMAX and XGBoost. At medium noise, LSTM remained one of the strongest models and was nearly tied with context-aware stacking. This confirms that recurrent sequence models are most useful when predictive information depends on persistent temporal state.

4.4 Regime-Switching Regime

The regime-switching process combined a low-volatility linear state with a higher-volatility nonlinear state. This was the main test of adaptive stacking. Validation performance for context-aware stacking was often strong, but out-of-sample test performance remained inconsistent. At medium noise, context-aware stacking reached test F1 of 0.5386, while logistic stacking and LSTM reached 0.6851 and 0.6526. Sample-size sensitivity experiments showed that the adaptive stacker could improve in some settings but could also become unstable, especially when the effective meta-training sample was limited.

4.5 Synthetic Summary

The synthetic experiments support four conclusions:
1. LSTM is strongest when temporal memory matters.
2. XGBoost is useful when threshold-type nonlinearity matters.
3. ARIMAX is a meaningful linear baseline but is not always best for final directional classification.
4. Stacking can help in controlled heterogeneous settings, but adaptive stacking is less stable than simpler stacking.

Table 2. Synthetic experiment results under medium-noise conditions across linear, nonlinear, sequential, and regime-switching regimes.

Regime: Linear
XGBoost: test accuracy 0.4465, test F1 0.2171, observation: weak performance in nominally linear setting
ARIMAX: test accuracy 0.4202, test F1 0.1433, observation: did not dominate final classification task
LSTM: test accuracy 0.4101, test F1 0.0068, observation: very poor in this setting
Logistic Stacking: test accuracy 0.5596, test F1 0.6746, observation: best-performing model in medium-noise linear regime
Context-Aware Stacking: test accuracy 0.5374, test F1 0.6617, observation: strong, but slightly below logistic stacking

Regime: Nonlinear
XGBoost: test accuracy 0.5071, test F1 0.5643, observation: competitive nonlinear learner
ARIMAX: test accuracy 0.5374, test F1 0.6336, observation: surprisingly strong despite nonlinear regime
LSTM: test accuracy 0.4667, test F1 0.4359, observation: weaker than tree and ensemble methods
Logistic Stacking: test accuracy 0.5838, test F1 0.6720, observation: best-performing model in medium-noise nonlinear regime
Context-Aware Stacking: test accuracy 0.4848, test F1 0.2975, observation: underperformed substantially

Regime: Sequential
XGBoost: test accuracy 0.4428, test F1 0.0652, observation: poor in sequential-memory setting
ARIMAX: test accuracy 0.5162, test F1 0.6267, observation: reasonable but not best
LSTM: test accuracy 0.6263, test F1 0.7102, observation: strongest base learner as expected
Logistic Stacking: test accuracy 0.6328, test F1 0.6840, observation: strong ensemble performance
Context-Aware Stacking: test accuracy 0.6350, test F1 0.7091, observation: nearly tied with LSTM

Regime: Regime Switching
XGBoost: test accuracy 0.4828, test F1 0.0791, observation: weak under changing regimes
ARIMAX: test accuracy 0.5253, test F1 0.5011, observation: stable but moderate
LSTM: test accuracy 0.5354, test F1 0.6526, observation: strong under regime persistence
Logistic Stacking: test accuracy 0.5616, test F1 0.6851, observation: best-performing model in medium-noise switching regime
Context-Aware Stacking: test accuracy 0.5051, test F1 0.5386, observation: improved but still less stable than logistic stacking

5. Experiments on Real Data

5.1 Dataset

The real-data experiment uses SPY from 2013-01-01 to 2024-01-01, downloaded using yfinance. Adjusted close is used as the main price series and volume is used for additional features. The data are chronologically sorted and cleaned before feature construction.

5.2 Experimental Setup

The chronological split is:
Train: 2013-01-01 to 2021-12-31
Validation: 2022-01-01 to 2022-12-31
Test: 2023-01-01 to 2024-01-01

Base models are trained on the train set. Validation predictions are used to train the stackers. The test set is reserved strictly for final evaluation.

5.3 Results

Validation results:
XGBoost: accuracy 0.4741, F1 0.6185
ARIMAX: accuracy 0.5020, F1 0.5211
LSTM: accuracy 0.4661, F1 0.6359
Logistic Stacking: accuracy 0.5339, F1 0.1583
Context-Aware XGBoost Stacking: accuracy 0.6375, F1 0.5517

Test results:
XGBoost: accuracy 0.5551, precision 0.6146, recall 0.8077, F1 0.6981, Sharpe 3.4872, cumulative return 1.5149
ARIMAX: accuracy 0.4286, precision 0.5889, recall 0.3397, F1 0.4309, Sharpe 2.2593, cumulative return 0.5494
LSTM: accuracy 0.6367, precision 0.6367, recall 1.0000, F1 0.7781, Sharpe 4.1685, cumulative return 2.1429
Logistic Stacking: accuracy 0.3878, precision 0.8750, recall 0.0449, F1 0.0854, Sharpe 2.1861, cumulative return 0.2177
Context-Aware XGBoost Stacking: accuracy 0.4122, precision 0.7143, recall 0.1282, F1 0.2174, Sharpe 2.8530, cumulative return 0.4657

5.4 Interpretation

LSTM is the strongest held-out model on SPY, both in classification and financial terms. This indicates that short-horizon directional information in SPY is better captured through temporal sequence modeling than through either a linear econometric model or a static tabular learner alone.

XGBoost is also competitive, which confirms that the technical indicator feature space contains useful nonlinear predictive structure. ARIMAX is weaker in classification performance but still generates positive trading statistics, which shows that financial usefulness and classification accuracy are not always identical.

The ensemble results are mixed. Logistic stacking performs poorly on the SPY test set despite showing some benefits in synthetic settings. The context-aware stacker improved meaningfully after simplification and regularization. In particular, reducing the context space to vol10, rsi, and trend_signal improved test recall, test F1, and cumulative return relative to earlier, more overfit versions. However, the final adaptive stacker still does not outperform LSTM or XGBoost on held-out SPY data.

This is not a contradiction of the project goal. Instead, it is a substantive result. The synthetic experiments show that ensemble learning can help when heterogeneous structure is present and controlled. The real-data experiment shows that robust adaptive stacking is harder to train than expected in financial settings because the meta-learning stage is itself noisy, regime-dependent, and data-limited.

Table 3. Held-out test results for the real-data SPY forecasting experiment.

XGBoost: accuracy 0.5551, precision 0.6146, recall 0.8077, F1 0.6981, Sharpe ratio 3.4872, cumulative return 1.5149
ARIMAX: accuracy 0.4286, precision 0.5889, recall 0.3397, F1 0.4309, Sharpe ratio 2.2593, cumulative return 0.5494
LSTM: accuracy 0.6367, precision 0.6367, recall 1.0000, F1 0.7781, Sharpe ratio 4.1685, cumulative return 2.1429
Logistic Stacking: accuracy 0.3878, precision 0.8750, recall 0.0449, F1 0.0854, Sharpe ratio 2.1861, cumulative return 0.2177
Context-Aware XGBoost Stacking: accuracy 0.4122, precision 0.7143, recall 0.1282, F1 0.2174, Sharpe ratio 2.8530, cumulative return 0.4657

6. Discussion

The strongest aspect of this project is the connection between theory and empirical evaluation. The synthetic experiments isolate model inductive bias under known structure, while the SPY experiment tests whether those conclusions remain meaningful in a real financial time series.

The results show that no single model class is universally best. Instead, model performance depends on the data-generating structure. LSTM performs best when predictive structure is sequential. XGBoost is effective for nonlinear interactions. ARIMAX provides an interpretable linear baseline. Stacking can improve performance in controlled heterogeneous settings.

At the same time, the project shows that adaptive stacking is difficult to train robustly on real financial data. Both synthetic and real-data experiments show that the context-aware stacker can fit validation regimes strongly but generalize inconsistently. The simplified final stacker is clearly better than earlier overfit versions, but it still remains weaker than the best base learner on SPY.

This does not weaken the project. On the contrary, it strengthens the methodological conclusion. The project demonstrates that meta-learning in finance is itself a regime-sensitive prediction problem. If the validation window is narrow or the context representation is too flexible, the adaptive ensemble can become more fragile than the single models it is trying to improve upon.

The project has several limitations. First, the real-data analysis focuses only on SPY, so conclusions may not generalize directly to more volatile individual stocks. Second, the feature set is restricted to technical and volume-based variables. Third, the financial backtest ignores transaction costs, slippage, and risk constraints. Finally, the meta-learning framework would likely benefit from more robust walk-forward or out-of-fold temporal training rather than a single validation-year approach.

7. Conclusion

This project investigated five-day stock direction forecasting using a heterogeneous ensemble framework composed of ARIMAX, XGBoost, LSTM, logistic stacking, and context-aware stacking.

The synthetic experiments showed that model inductive bias matters. LSTM performed best in sequential latent-state regimes, XGBoost was effective in nonlinear settings, ARIMAX remained a useful linear baseline, and stacking often improved performance under controlled heterogeneous structure. However, adaptive stacking was generally less stable than simpler stacking.

The real-data experiments on SPY showed that LSTM was the strongest held-out model, with XGBoost as a competitive alternative and ARIMAX as an interpretable benchmark. The final context-aware stacker improved after regularization, threshold tuning, and reduction of the context feature set, but it still did not outperform the best base learner on held-out SPY data.

The final conclusion is therefore more nuanced than a simple claim that ensemble methods always win. Ensemble learning can help when model skill varies across controlled regimes, but robust adaptive stacking is difficult to deploy reliably on noisy real financial data. In this project, that difficulty became one of the main findings rather than a failure of the study. The project therefore demonstrates not only model implementation, but also a careful empirical investigation of when adaptive ensemble learning helps and when it fails to generalize.

References

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. Time Series Analysis: Forecasting and Control.
2. Chen, T., and Guestrin, C. XGBoost: A Scalable Tree Boosting System.
3. Hochreiter, S., and Schmidhuber, J. Long Short-Term Memory.
4. Wolpert, D. Stacked Generalization.
5. Documentation for statsmodels, xgboost, tensorflow, and yfinance.
