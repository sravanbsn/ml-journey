# House Prices Linear Regression

## Overview
Built a linear regression model completely from scratch using only NumPy and Pandas — no Scikit-learn or any ML libraries. Trained on the Kaggle House Prices dataset to predict sale prices from living area.

## What I implemented
- Hypothesis function: `h(x) = wx + b`
- MSE cost function from scratch
- Gradient descent optimisation loop
- Min-max normalisation for both X and y
- Denormalisation to convert predictions back to real prices
- Loss curve visualisation
- Hypothesis line plotted over actual data

## Dataset
Kaggle — House Prices: Advanced Regression Techniques
- 1460 houses
- Features used: GrLivArea (living area in sqft)
- Target: SalePrice

## Results
- Initial cost (w=0, b=0): ~39 billion
- Final cost after 1000 epochs: ~0.05 (normalised)
- Predicted price for 2000 sqft house: $190,104

## Key learnings
- Why normalisation is essential for gradient descent to converge
- How learning rate affects training speed and stability
- Why one feature is not enough — outliers affect the hypothesis line significantly
- The complete ML pipeline from raw data to prediction

## Tech stack
- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## Files
- `house_prices_linear_regression.ipynb` — main notebook with all code and visualisations
- `train.csv` — Kaggle dataset (not uploaded — download from Kaggle)

## Next steps
- Add multiple features (bedrooms, location, age)
- Implement polynomial regression
- Compare with Scikit-learn LinearRegression
