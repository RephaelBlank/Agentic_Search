# colab-freshqa-experiments

## Overview
This project provides a framework for running experiments with the FreshQA evaluation pipeline. It allows users to evaluate various models and prompt types against a set of questions, generating reports based on the results. The project also includes a placeholder for future fine-tuning experiments.

This small project contains a Colab-ready notebook and minimal runner/judge placeholders.

## Quick start in Colab
1. Upload this repo to Colab or `git clone` it into /content.
2. (Optional) Uncomment the `!pip install -r .../requirements.txt` line in the first notebook cell.
3. Mount Google Drive if your CSV data is stored there.
4. Run the notebook cells. The notebook uses `src/runner.py` and `src/judge.py` placeholders to produce `results.json` and `judgements.json`.

## Project Structure
```
colab-freshqa-experiments
├── notebooks
│   ├── run_experiments.ipynb  # Jupyter notebook for running experiments
│   └── fine_tune.ipynb        # Jupyter notebook for fine-tuning experiments (to be developed)
├── src
│   ├── runner.py               # Main logic for executing the FreshQA evaluation pipeline
│   ├── agent.py                # Functions for executing web searches and interacting with the Tavily API
│   ├── judge.py                # Logic for evaluating model answers against expected answers
│   └── utils.py                # Utility functions for supporting main functionalities
├── data
│   └── freshqa_filtered.csv     # CSV file containing filtered questions and answers
├── .env.example                 # Template for environment variables
├── requirements.txt             # List of Python dependencies required for the project
└── README.md                    # Documentation for the project
```

## Setup Instructions
1. **Clone the Repository**
   Clone this repository to your local machine or Google Colab environment.

2. **Install Dependencies**
   Install the required Python packages listed in `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   Copy `.env.example` to `.env` and fill in the necessary API keys and configurations.

4. **Running Experiments**
   Open `notebooks/run_experiments.ipynb` to select a model, prompt type, and judge model. Execute the cells to run the experiments and generate reports.

5. **Fine-Tuning (Future Development)**
   The `notebooks/fine_tune.ipynb` is a placeholder for future fine-tuning experiments. This notebook will be developed later to include functionalities for model fine-tuning.

## Usage Guidelines
- Use the provided Jupyter notebooks to interact with the evaluation pipeline.
- Modify the parameters in `run_experiments.ipynb` to test different models and prompts.
- Review the generated reports to analyze the performance of the models.

## To push code to GitHub
- Add the files and commit. Example:
  ```
  git add .
  git commit -m "Add Colab-ready notebook and minimal runner/judge"
  git push origin main
  ```

## Contributing
Contributions are welcome! Please feel free to submit issues or pull requests for improvements or additional features.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.