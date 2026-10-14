# Electric Vehicle Adoption: Exploratory Analysis and Powertrain Classification

Dissertation project on electric vehicle (EV) adoption in the US. This repository holds the data analysis and modelling notebook ([`dissertation.ipynb`](dissertation.ipynb)) for the vehicle registration part of the work.

## Data

Washington State Department of Licensing [Electric Vehicle Population Data](https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2): 100,975 registered EVs with 17 attributes, including location (county, city, census tract, coordinates), make and model, model year, electric range, base MSRP, electric utility and Clean Alternative Fuel Vehicle (CAFV) eligibility. The CSV is not included; download it from the link and save it next to the notebook as `1.csv`.

## What the notebook does

1. **Cleaning.** Drops the two rows with no county or city, imputes the remaining gaps (median for numeric, mode for categorical), and parses vehicle coordinates out of the WKT `POINT` field.
2. **Exploratory analysis.** EV counts by county, model, electric utility and CAFV eligibility; electric range and model year by CAFV eligibility; feature correlations.
3. **Classification.** Predicts whether a vehicle is a battery electric vehicle (BEV, 77%) or a plug-in hybrid (PHEV, 23%).
   - Encoding: ordinal for low-cardinality fields, frequency encoding for high-cardinality ones (county, city, model, utility, VIN prefix).
   - Class imbalance handled with NearMiss undersampling on the training split only.
   - Models: Gaussian Naive Bayes, Logistic Regression, Random Forest and XGBoost, each tuned with grid search under 10-fold cross-validation.

## Results

| Model | Test accuracy |
|---|---|
| XGBoost | 0.9998 |
| Random Forest | 0.9997 |
| Naive Bayes | 0.767 |
| Logistic Regression | 0.519 |

The most important Random Forest features were electric range (0.41), model (0.18), CAFV eligibility (0.17), make (0.08) and VIN prefix (0.06).

## Limitations (read before trusting the numbers above)

- **The task is close to a lookup.** Model, VIN prefix and electric range nearly determine whether a vehicle is a BEV or a PHEV, and CAFV eligibility is itself defined from range and powertrain. The near-perfect tree-model scores say more about these features than about the models. A more informative version would drop vehicle-identity features and ask what location, utility and price reveal about powertrain choice.
- **Test scores were computed with cross-validation on the test split.** The models were refit on folds of the test data rather than trained once and scored on the held-out set, so the test column is not a true held-out estimate.
- **Linear models were not scaled.** Logistic Regression ran on unscaled, frequency-encoded features, which likely explains its near-chance score.
- **Air quality analysis is not in this repository.** The dissertation's wider question, how EV adoption relates to air quality, is covered in the written dissertation. Its conclusion was that the air quality index depended less on EV adoption than on other pollution sources.

## Run it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn category-encoders missingno
jupyter notebook dissertation.ipynb
```

Or open it in Colab with the badge at the top of the notebook.
