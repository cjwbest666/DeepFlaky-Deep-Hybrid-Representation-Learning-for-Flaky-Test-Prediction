# DeepFlaky-Deep-Hybrid-Representation-Learning-for-Flaky-Test-Prediction
DeepFlaky is a novel framework that accurately predicts flaky tests by fusing expert-defined features with deep semantic features from CodeBERT, and enhancing the representation through contrastive learning. 

## Project Overview
This project is the code implementation corresponding to the paper "DeepFlaky Deep Hybrid Representation Learning for Flaky Test Prediction". The paper proposes a novel framework for detecting flaky tests. By fusing expert-defined features with deep semantic features from CodeBERT, and enhancing the representation through contrastive learning, it improves the precision of detection.

## Environment Requirements
The dependencies required to run the project can be installed using the `requirements.txt` file. Use the following command:

```bash
pip install -r requirements.txt
```

## Usage
To run the project, you can follow these steps:
1. Clone the repository:
2. Ensure you have the necessary configurations set in `DeepFlaky-RQ1.py`, `DeepFlaky-RQ2.py`, `DeepFlaky-RQ3-contrastive learning.py` , `DeepFlaky-RQ3-without contrastive learning.py` and `DeepFlaky-RQ5.py`. You may need to adjust paths or parameters according to your environment and dataset.
3. Execute the script:
   ```bash
   python `DeepFlaky-RQ1.py`
   python `DeepFlaky-RQ2.py`
   python `DeepFlaky-RQ3-contrastive learning.py`
   python `DeepFlaky-RQ3-without contrastive learning.py`
   python `DeepFlaky-RQ5.py`
   ```
4. The results will be saved in the `result-RQ1`, `result-RQ2` `result-RQ3` and `result-RQ5`directory.

## Project Structure
```
├── dataset_for_semantic_extract/FlakeFlagger
├── input_data/
│   ├── FlakeFlaggerFeature.csv
│   └── FlakeFlaggerFeaturesTypes.csv
├── original_tests/
├── Predicting_with_LLaMA/
│   ├── Predicting_flaky_tests_with_llama_few_shot.py
│   └── Predicting_flaky_tests_with_llama_zero_shot.py
├── result-RQ1/
├── result-RQ2/
├── result-RQ3/
├── result-RQ5/
├── result/
├── Semantic_Representation_Extraction.py
├── README.md
├── requirements.txt
├── DeepFlaky-RQ1.py
├── DeepFlaky-RQ2.py
├── DeepFlaky-RQ3-contrastive learning.py
└── DeepFlaky-RQ3-without contrastive learning.py
└── DeepFlaky-RQ5
```

