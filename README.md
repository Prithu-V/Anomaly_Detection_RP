# A Hybrid Unsupervised Learning Framework for Multi-Type Anomaly Detection in Database Query Logs

Research project by **Prithu Verma** and **Niranjan Lal**  
Department of Computer Science and Engineering, SRM Institute of Science and Technology, Delhi-NCR Campus

## Overview

Database query logs contain signals about how users and applications interact with a database. Unusual access patterns, unexpected query activity outside normal hours, and abnormal execution times can indicate security or operational problems.

This project investigates whether unsupervised machine learning can detect three types of database-query anomalies within one framework:

- **Security anomalies** — queries that access a store outside a user's expected jurisdiction.
- **Temporal anomalies** — query activity outside the simulated working schedule, including overnight and weekend activity.
- **Performance anomalies** — queries whose execution time is substantially higher than the expected baseline for their query type.

The experiments compare three unsupervised models trained on normal query records: **Isolation Forest**, **DBSCAN**, and **One-Class Support Vector Machine (One-Class SVM)**. The study also evaluates a union ensemble that flags a query as anomalous when either DBSCAN or One-Class SVM flags it.

> **Research scope:** The experiments use a synthetically generated dataset built around the Pagila PostgreSQL sample database. The results are not a production deployment evaluation.

## Research question

Can a single unsupervised anomaly-detection model identify security, temporal, and performance anomalies effectively, or is a combination of models needed to cover different anomaly types?

## Key findings

The dataset contains **10,000 query-log records**, of which **7.25% (725 records)** are labelled anomalies. The remaining 9,275 records represent normal activity. Labels were retained for evaluation and were not supplied to the models during training or prediction.

### F1 score by anomaly type

| Model | Security | Temporal | Performance |
|---|---:|---:|---:|
| Isolation Forest | 0.032 | 0.046 | 0.238 |
| DBSCAN | 0.714 | 0.972 | 0.528 |
| One-Class SVM | 0.491 | 0.535 | 0.634 |

### Aggregate model results

| Model | Precision | Recall | F1 score |
|---|---:|---:|---:|
| Isolation Forest | 0.20 | 0.23 | 0.22 |
| DBSCAN | 1.00 | 0.58 | 0.74 |
| One-Class SVM | 0.66 | 1.00 | 0.79 |
| DBSCAN + One-Class SVM (union) | 0.658 | 1.00 | 0.794 |

**Interpretation**

- **DBSCAN** performed best on temporal anomalies (F1 = 0.972) and security anomalies (F1 = 0.714).
- **One-Class SVM** performed best on performance anomalies (F1 = 0.634) and achieved recall of 1.00 in the reported experiment.
- **Isolation Forest** performed poorly in this feature setup, particularly with the high-dimensional encoded feature matrix.
- The **union ensemble** detected all 725 labelled anomalies in this experiment (recall = 1.00), but had lower precision than DBSCAN. This means it avoided false negatives in the test dataset at the cost of more false positives.

These findings support evaluating anomaly-specific performance instead of relying only on one overall metric. They do not establish that the same results will hold on real production logs.

## Methodology

### 1. Dataset construction

The Pagila PostgreSQL sample database was used as the structural basis for the experiment. Two additional tables supported the simulation:

- `users` — represents database users, roles, privileges, and store jurisdiction.
- `query_log` — stores a record for each simulated query execution.

The synthetic workload covers 14 simulated days. Normal traffic is restricted to weekdays from 09:00 to 18:00 and includes `SELECT`, `INSERT`, `UPDATE`, and `DELETE` query types in a read-heavy workload.

The anomaly labels were assigned during synthetic data generation. They were not included as model input.

### 2. Feature engineering

The feature set represents query behaviour, access patterns, and execution characteristics. The paper describes the following raw and engineered signals:

- User identity and accessed store
- Query type and user role
- Execution time and rows returned
- Query frequency during the preceding 60 seconds
- Hour of day and day of week
- `is_off_hours`, `is_weekend`, and `is_cross_store`
- `execution_time_zscore`, calculated relative to the relevant query-type baseline
- `query_risk_score`, representing the sensitivity of a query type

Categorical features were one-hot encoded, and numerical features were standardized with `StandardScaler`. The full encoded feature matrix contained 35 features. A reduced 12-feature matrix was also tested for Isolation Forest because its initial results were poor in the high-dimensional representation.

### 3. Models

| Model | Role in the experiment | Configuration reported in the paper |
|---|---|---|
| Isolation Forest | Isolates observations that are easier to separate from normal behaviour | `n_estimators=100`, `contamination=0.0725`, `random_state=42` |
| DBSCAN | Identifies low-density observations as noise | `eps=0.5`, `min_samples=5` |
| One-Class SVM | Learns a boundary around normal behaviour | RBF kernel, `nu=0.0725`, `gamma="scale"` |
| Union ensemble | Flags an observation if DBSCAN **or** One-Class SVM flags it | Logical OR of the two model predictions |

All models were trained using the 9,275 normal records. Evaluation used the synthetic ground-truth labels.

### 4. Evaluation

The study reports precision, recall, and F1 score, both overall and separately for each anomaly type. Per-type F1 scores are important because a strong result for one anomaly class can conceal poor performance on another.

Accuracy was treated as a secondary metric because the dataset is imbalanced: a classifier that labels every record as normal would still achieve 92.75% accuracy while missing all anomalies.

## Technology stack

The paper reports the following environment:

- **Database:** PostgreSQL 14
- **Language:** Python 3.12
- **Machine learning:** scikit-learn 1.6
- **Data manipulation:** pandas 2.2
- **Numerical computing:** NumPy 1.26
- **Experiment environment:** Google Colaboratory
- **Development hardware:** MacBook Pro, Intel Core i5 at 2.3 GHz, 16 GB RAM

## Repository contents

The exact filenames depend on how the experiments are organized in this repository. A typical layout for this project could be:

```text
.
├── README.md
├── Research Paper.pdf
├── notebooks/
│   └── anomaly_detection_experiments.ipynb
├── scripts/
│   ├── generate_dataset.py
│   ├── feature_engineering.py
│   └── train_and_evaluate.py
└── results/
    ├── metrics.csv
    └── figures/
```

This is a suggested layout, not a claim about the files currently present in the repository. Adjust it to match the actual repository contents.

## Reproducing the experiments

The paper describes the dataset-generation and model-evaluation approach, but the steps below are a general setup guide. The exact commands depend on the scripts and notebooks included in the repository.

### Prerequisites

- Python 3.12
- PostgreSQL 14
- A Pagila database instance, if using the database-backed data-generation workflow
- The experiment notebooks or scripts from this repository

### 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-DIRECTORY>
```

Replace the placeholders with your actual GitHub repository URL and directory name.

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install the core Python dependencies

```bash
pip install scikit-learn==1.6.0 pandas==2.2.0 numpy==1.26.0
```

These versions reflect the versions stated in the paper. The exact versions used in the original environment may differ by patch release; use a repository-provided dependency file if one is available.

### 4. Configure PostgreSQL and the dataset

Install and configure PostgreSQL 14, load Pagila, and follow the repository's data-generation instructions to create the synthetic `users` and `query_log` data. Do not assume that the dataset is included unless it is present in the repository.

### 5. Run the experiments

Use the notebooks or scripts actually included in the repository to run feature engineering, model training, evaluation, and plotting. Add the exact commands here once the repository's filenames and execution order have been confirmed.

## Limitations

- The workload and anomaly labels are synthetic, so the experiment may not capture the complexity of real database traffic.
- The dataset contains 10,000 records and uses a controlled anomaly rate of 7.25%.
- Model results depend on the chosen features, preprocessing, and hyperparameters.
- The union ensemble prioritizes recall; its false-positive rate and operational cost would need careful evaluation before deployment.
- The framework has not been validated on production database logs in the reported study.

## Future work

The paper identifies several directions for further research:

- Evaluate the framework on real production query logs.
- Test additional unsupervised algorithms.
- Investigate more advanced ensemble methods.
- Assess whether the results generalize across workloads and database environments.

## Paper

The full research paper is included in this repository as `Research Paper.pdf` (update this filename if the uploaded paper has a different name in your GitHub repository).

**Title:** *A Hybrid Unsupervised Learning Framework for Multi-Type Anomaly Detection in Database Query Logs*  
**Authors:** Prithu Verma and Niranjan Lal

## License

No license is specified here. Add a license file and update this section if you intend to grant others permission to use, modify, and distribute the code.

## Acknowledgements

The experiments use the Pagila PostgreSQL sample database as the structural basis for synthetic query-log generation.
