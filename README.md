# Real-Time Score Prediction using Machine Learning

## Description
This project employs machine learning techniques to make real-time predictions of football match scores. By using Python and linear regression, the project processes match data and predicts the likelihood of a team winning, losing, or drawing. The dataset includes matches from the English Premier League, covering multiple seasons.

## Tech Stack
- **Python**: Programming language used for implementing the project.
- **Pandas**: Data manipulation and analysis library.
- **Scikit-learn**: Machine learning library for Python.

## File Structure
- `.gitignore`: Specifies files to be ignored by Git.
- `LICENSE`: License for the project.
- `Prediction.ipynb`: Jupyter notebook containing the prediction model and analysis.
- `README.md`: Project documentation.
- `code.py`: Python script for data preprocessing and model training.
- `logo.jpeg`: Project logo.
- `matches.csv`: Dataset containing match details.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## How to Run
1. Ensure you are in the project directory.
2. Run the Jupyter notebook:
    ```sh
    jupyter notebook Prediction.ipynb
    ```
3. Alternatively, you can run the Python script:
    ```sh
    python code.py
    ```

## Usage
- **Data Preprocessing**: The `code.py` script reads the `matches.csv` file, preprocesses the data, and prepares it for model training.
- **Model Training**: The Random Forest classifier is trained on historical match data to predict future match outcomes.
- **Prediction**: The trained model predicts match outcomes for the test dataset.

## Possible Output
The output of the model includes:
- Predicted match outcomes (win, lose, draw) for each match in the test dataset.
- Accuracy and precision scores to evaluate the model's performance.

## Example
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Load data
matches = pd.read_csv('matches.csv', index_col=0)

# Preprocess data
matches['date'] = pd.to_datetime(matches['date'])
matches['target'] = (matches['result'] == 'W').astype('int')
matches['venue_code'] = matches['venue'].astype('category').cat.codes
matches['opp_code'] = matches['opponent'].astype('category').cat.codes
matches['hour'] = matches['time'].str.replace(":.+", "", regex=True).astype('int')
matches['day_code'] = matches['date'].dt.dayofweek

# Train-test split
train = matches[matches['date'] < '2022-01-01']
test = matches[matches['date'] > '2022-01-01']

# Model training
predictors = ['venue_code', 'opp_code', 'hour', 'day_code']
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
rf.fit(train[predictors], train['target'])

# Prediction
preds = rf.predict(test[predictors])

# Evaluation
precision = precision_score(test['target'], preds)
print(f'Precision: {precision:.2f}')
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
