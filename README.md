 # Emotion Sentiment Classifier

 A machine learning project for classifying emotions in tweets, leveraging LSTM neural networks. It is able to detect 6 emotions categories: sadness, joy, love, anger, fear and surprise.

 ## Features

 - Automatic preprocessing of tweets (stop word and punctuation removal).
 - Emotion classification using Sequential Long Short Term Memory (LSTM) model.
 - Support for analysing large data sets (>400,000 entries).

## Installation

Install Python 3.11.11 (if not already installed). [See Platform Compatibility](#platform-compatibility)

    brew install python@3.11

Once installed set up the Python virtual environment with the following:

    mkdir my_venv
    cd my_venv
    python3.11 -m my_venv .

To activate your virtual environment, type the following into the command line:

    source bin/activate

To ensure your virtual environment is using Python 3.11, run the following:

    python3 -V

Your output should be:

    Python 3.11.11

Some external libraries used may not support earlier or later version on Python.
### External Libraries

Several external libraries are required and must be installed before running the program. A requirements.txt file is included, to install the requirements either:

Type the following into the terminal (this is recommended):

    pip3 install -r requirements.txt

Or if you prefer, you can install each library individually with the following:

    pip3 install nltk tensorflow scikit-learn pandas seaborn matplotlib
    
## Usage

Prepare your dataset in CSV format with three columns: index, tweet and emotion label. The first line should be a header, for example:

    ,text,label
    0,i just feel really helpless and heavy hearted,4


To run the script, run the following command:

    python3 emotion_sentiment_classifier.py <your dataset.csv>

Preprocessing, training and evaluation are all run, with appropriate outputs given after each step.

## Dataset

- Dataset used for training and testing: Tweets classified into six emotion catergories.
- Source: https://www.kaggle.com/datasets/nelgiriyewithana/emotions
- Format: CSV with 3 columns: index, text and emotion label.

## Additional Notes

- Ensure dataset matches required format.
- Libraries and tools require specific Python version for compatibility.

## Platform Compatibility

> Please note: this project was developed on macOS and bash. All commands and setup instructions provided are tailored to macOS bash environments. If you are using other platforms, commands given may need to be adjusted accordingly.