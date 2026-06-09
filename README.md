# NBA Player Longevity — Feature Engineering Pipeline

## Project Overview
This project builds a clean, model-ready dataset from raw NBA rookie statistics to predict **career longevity**: whether a player will last **5 or more seasons** in the NBA.

## Dataset
| Property | Detail |
|----------|--------|
| Source | NBA Players Statistics (simulated from historical distributions) |
| Samples | 1,340 rookie seasons |
| Raw Features | 20 (counting stats, shooting percentages, game logs) |
| Target | `target_5yrs` — binary (1 = survived 5+ seasons, 0 = did not) |
| Class Balance | 51.3% / 48.7% — near-balanced, no resampling needed |

## Prediction Goal
Classify NBA rookies as likely to have long careers (≥ 5 years) or short careers (< 5 years) based on their first-season performance statistics. This has real-world applications in player scouting, contract valuation, and roster building.

## Feature Engineering Workflow

### Step 1 — Target Definition
`target_5yrs` confirmed as clean binary variable with no missing values.

### Step 2 — Drop Non-Predictive Columns
`Name` removed — string identifier with no predictive signal; retaining it risks data leakage.

### Step 3 — Correlation Analysis (Multicollinearity Reduction)
6 pairs with |r| > 0.85 identified. For each pair, the feature with weaker individual correlation to the target was removed:

| Dropped | Retained | Reason |
|---------|----------|--------|
| `PTS` | `FGM` | PTS collinear with FGM; FGM had stronger target correlation |
| `FGA` | `FGM` | FGA collinear with FGM |
| `FTM` | `FTA` | FTM ≈ FTA × FT%; FTA retained |
| `DREB` | `REB` | DREB is a component of REB |
| `3PA` | `3P Made` | Volume vs. makes; 3P Made slightly stronger signal |

### Step 4 — Feature Engineering (3 composite metrics)
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `EFF_RATING` | (FGM + REB + AST + STL + BLK − missed FTs − TOV) / GP | NBA's official efficiency metric per game |
| `AST_TO_RATIO` | AST / (TOV + 0.001) | Decision-making quality — key longevity predictor |
| `REB_PER_MIN` | REB / (MIN + 0.001) | Rebounding intensity adjusted for playing time |

### Step 5 — Missing Value Handling
~3% nulls in PTS, REB, AST (plus derived features). Resolved via **median imputation** — robust to outliers, preserves all 1,340 rows.

## Final Dataset
- **1,340 samples · 17 features + 1 target**
- Zero missing values
- Ready for Logistic Regression, Random Forest, XGBoost, or any sklearn classifier

## Project Structure
```
nba-feature-engineering/
├── nba_feature_engineering.ipynb  # Main notebook (all cells executed)
├── nba_players.csv                # Raw dataset
├── nba_features_clean.csv         # Output — clean, model-ready dataset
├── README.md                      # This file
├── requirements.txt               # Dependencies
├── target_distribution.png        # EDA — class balance
├── correlation_heatmap_full.png   # Full correlation matrix
├── target_correlations.png        # Feature-target correlations
├── engineered_features.png        # New feature distributions
└── final_feature_importance.png   # Final feature signal chart
```

## Setup
```bash
pip install -r requirements.txt
jupyter notebook nba_feature_engineering.ipynb
```

## Author
**Abubakar Jibrin Gunda**  
AI/Data Professional · Kano State, Nigeria
