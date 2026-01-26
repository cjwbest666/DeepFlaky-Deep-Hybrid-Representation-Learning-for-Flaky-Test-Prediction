import os
import json
import time
import requests
import numpy as np
from typing import List, Dict, Tuple
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class LlamaFlakyPredictor:
    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "llama3:8b"):
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api/generate"

        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags")
            print(f"Successfully connected to Ollama")
        except Exception as e:
            print(f"Unable to connect to Ollama: {e}")
            print("Please ensure Ollama is running: ollama serve")
            exit(1)

    def read_java_test_file(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"Failed to read file {file_path}: {e}")
            return ""

    def construct_prompt(self, java_code: str) -> str:
        prompt = f"""You are a software testing analysis expert. Please carefully analyze the following Java test code and determine whether it is a "flaky test" (unstable test).

    Flaky test definition: A test that sometimes passes and sometimes fails when run multiple times with the same code and environment configuration.

    **Common characteristics of Flaky tests**:
    1. Concurrency/multithreading issues (e.g., race conditions)
    2. Insufficient waiting for asynchronous operations
    3. Dependency on external resources (network, file system, database)
    4. Use of random numbers or non-deterministic values
    5. Test execution order dependency
    6. Time dependency (e.g., Thread.sleep())
    7. Environment configuration dependency

    **Characteristics of Non-Flaky tests**:
    1. Completely deterministic behavior
    2. No dependency on external state
    3. Repeatable and consistent results
    4. No race conditions
    5. Self-contained tests

    Based on the above definitions, please analyze and:
    1. If the code shows clear Flaky characteristics, output "Flaky"
    2. If the code appears to be a stable and reliable test, output "Non-Flaky"
    3. Output only one of these two words, no additional content

    Java test code:
    ```java
    {java_code}
        Judgment result:"""
        return prompt

    ### Option 2: Improved response parsing

    def predict(self, java_code: str, max_retries: int = 3) -> str:
        prompt = self.construct_prompt(java_code)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 50
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=120)

                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "").strip()

                    # Clean response text
                    response_text = response_text.replace('"', '').replace("'", "")

                    # Extract prediction result
                    if "Flaky" in response_text:
                        return "Flaky"
                    elif "Non-Flaky" in response_text or "NonFlaky" in response_text:
                        return "Non-Flaky"
                    else:
                        return "Unknown"
                else:
                    print(f"API request failed (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2)

            except Exception as e:
                print(f"Request exception (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)

        return "Error"


    def process_single_file(self, file_path: str, true_label: str) -> Dict:
        """Process a single file and return results"""
        java_code = self.read_java_test_file(file_path)
        if not java_code:
            return None

        file_name = os.path.basename(file_path)
        prediction = self.predict(java_code)

        # Debug: log actual response
        print(f"  File: {file_name}")
        print(f"  True label: {true_label}")
        print(f"  Prediction: {prediction}")

        return {
            "file_name": file_name,
            "file_path": file_path,
            "true_label": true_label,
            "predicted_label": prediction,
            "is_correct": prediction == true_label
        }

    def process_project(self, project_path: str, project_name: str) -> Dict:
        """Process a single project"""
        print(f"Processing project: {project_name}")

        results = []
        stats = {
            "project_name": project_name,
            "total_files": 0,
            "flaky_files": 0,
            "nonflaky_files": 0
        }

        # Process flaky tests
        flaky_dir = os.path.join(project_path, "flakyMethods")
        if os.path.exists(flaky_dir):
            flaky_files = glob.glob(os.path.join(flaky_dir, "*.java"))
            for file_path in flaky_files:
                result = self.process_single_file(file_path, "Flaky")
                if result:
                    results.append(result)
                    stats["flaky_files"] += 1
                    stats["total_files"] += 1

                # Avoid making requests too quickly
                time.sleep(0.3)

        # Process non-flaky tests
        nonflaky_dir = os.path.join(project_path, "nonFlakyMethods")
        if os.path.exists(nonflaky_dir):
            nonflaky_files = glob.glob(os.path.join(nonflaky_dir, "*.java"))
            for file_path in nonflaky_files:
                result = self.process_single_file(file_path, "Non-Flaky")
                if result:
                    results.append(result)
                    stats["nonflaky_files"] += 1
                    stats["total_files"] += 1

                time.sleep(0.3)

        print(f"  Completed: processed {stats['total_files']} files")

        return {
            "results": results,
            "stats": stats
        }


class MetricsCalculator:
    """Metrics calculator"""

    @staticmethod
    def calculate_metrics_for_project(project_results: List[Dict]) -> Dict:
        """Calculate metrics for a single project"""
        if not project_results:
            return None

        # Extract labels
        true_labels = []
        pred_labels = []

        for result in project_results:
            true_labels.append(result["true_label"])
            pred_labels.append(result["predicted_label"])

        # Convert to binary (Flaky=1, Non-Flaky=0)
        true_binary = [1 if label == "Flaky" else 0 for label in true_labels]
        pred_binary = [1 if label == "Flaky" else 0 for label in pred_labels]

        # Calculate confusion matrix
        tp = fp = tn = fn = 0

        for true, pred in zip(true_binary, pred_binary):
            if true == 1:
                if pred == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred == 1:
                    fp += 1
                else:
                    tn += 1

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        # Attempt to calculate AUC (requires prediction probabilities, using simple approximation here)
        # Note: Since we only have category predictions without probabilities, we use predicted labels as an approximation
        try:
            auc = roc_auc_score(true_binary, pred_binary)
        except:
            auc = None

        return {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "auc": auc if auc is not None else "N/A",
            "total_samples": len(project_results)
        }

    @staticmethod
    def calculate_overall_metrics(all_project_metrics: Dict[str, Dict]) -> Dict:
        """Calculate overall metrics"""
        # Combine confusion matrices from all projects
        total_tp = total_fp = total_tn = total_fn = 0
        total_samples = 0
        all_true_labels = []
        all_pred_labels = []

        for project_name, project_data in all_project_metrics.items():
            if "results" in project_data:
                for result in project_data["results"]:
                    all_true_labels.append(result["true_label"])
                    all_pred_labels.append(result["predicted_label"])
                    total_samples += 1

        # Calculate overall confusion matrix
        true_binary = [1 if label == "Flaky" else 0 for label in all_true_labels]
        pred_binary = [1 if label == "Flaky" else 0 for label in all_pred_labels]

        for true, pred in zip(true_binary, pred_binary):
            if true == 1:
                if pred == 1:
                    total_tp += 1
                else:
                    total_fn += 1
            else:
                if pred == 1:
                    total_fp += 1
                else:
                    total_tn += 1

        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (total_tp + total_tn) / total_samples if total_samples > 0 else 0

        try:
            auc = roc_auc_score(true_binary, pred_binary)
        except:
            auc = None

        return {
            "TP": total_tp,
            "FP": total_fp,
            "TN": total_tn,
            "FN": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "auc": auc if auc is not None else "N/A",
            "total_samples": total_samples,
            "total_projects": len(all_project_metrics)
        }


def save_results(all_results: Dict, project_metrics: Dict, overall_metrics: Dict):
    """Save results to files"""

    output_dir = "llama_results"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save detailed results
    detailed_results = {
        "overall_metrics": overall_metrics,
        "project_metrics": project_metrics,
        "detailed_results": all_results
    }

    with open(os.path.join(output_dir, "detailed_results.json"), 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    # 2. Save project metrics summary (CSV format)
    import csv

    csv_path = os.path.join(output_dir, "project_metrics.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Project", "TP", "FP", "TN", "FN", "Precision", "Recall", "F1-Score", "Accuracy", "AUC", "Samples"])

        for project_name, metrics in project_metrics.items():
            writer.writerow([
                project_name,
                metrics["TP"],
                metrics["FP"],
                metrics["TN"],
                metrics["FN"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["accuracy"],
                metrics["auc"],
                metrics["total_samples"]
            ])

    # 3. Save overall metrics
    summary_path = os.path.join(output_dir, "overall_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("LLAMA FLAKY TEST PREDICTION - OVERALL SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total Projects: {overall_metrics['total_projects']}\n")
        f.write(f"Total Samples: {overall_metrics['total_samples']}\n\n")

        f.write("CONFUSION MATRIX:\n")
        f.write(f"True Positives (TP): {overall_metrics['TP']}\n")
        f.write(f"False Positives (FP): {overall_metrics['FP']}\n")
        f.write(f"True Negatives (TN): {overall_metrics['TN']}\n")
        f.write(f"False Negatives (FN): {overall_metrics['FN']}\n\n")

        f.write("PERFORMANCE METRICS:\n")
        f.write(f"Precision: {overall_metrics['precision']}\n")
        f.write(f"Recall:    {overall_metrics['recall']}\n")
        f.write(f"F1-Score:  {overall_metrics['f1_score']}\n")
        f.write(f"Accuracy:  {overall_metrics['accuracy']}\n")
        f.write(f"AUC:       {overall_metrics['auc']}\n")

    print(f"Results saved to directory: {output_dir}")
    print(f"  - Detailed results: detailed_results.json")
    print(f"  - Project metrics: project_metrics.csv")
    print(f"  - Overall summary: overall_summary.txt")


def quick_test():
    """Quick test for a single project"""

    print("Quick test mode")
    predictor = LlamaFlakyPredictor()

    # Test a single project
    test_project = "activiti-activiti"
    project_path = os.path.join("dataset", test_project)

    if not os.path.exists(project_path):
        print(f"Project does not exist: {project_path}")
        return

    # Only test first 3 files
    print(f"Testing project: {test_project}")

    # Test flaky files
    flaky_dir = os.path.join(project_path, "flakyMethods")
    if os.path.exists(flaky_dir):
        flaky_files = glob.glob(os.path.join(flaky_dir, "*.java"))[:2]

        print(f"Testing {len(flaky_files)} flaky files:")
        for file_path in flaky_files:
            java_code = predictor.read_java_test_file(file_path)
            if java_code:
                prediction = predictor.predict(java_code)
                print(f"  {os.path.basename(file_path)}: Prediction={prediction}, Actual=Flaky")

    # Test non-flaky files
    nonflaky_dir = os.path.join(project_path, "nonFlakyMethods")
    if os.path.exists(nonflaky_dir):
        nonflaky_files = glob.glob(os.path.join(nonflaky_dir, "*.java"))[:2]

        print(f"\nTesting {len(nonflaky_files)} non-flaky files:")
        for file_path in nonflaky_files:
            java_code = predictor.read_java_test_file(file_path)
            if java_code:
                prediction = predictor.predict(java_code)
                print(f"  {os.path.basename(file_path)}: Prediction={prediction}, Actual=Non-Flaky")


def main():
    print("=" * 50)
    start_time = time.time()
    # Initialize predictor
    predictor = LlamaFlakyPredictor()
    calculator = MetricsCalculator()

    # Set dataset directory
    dataset_root = "dataset"

    if not os.path.exists(dataset_root):
        print(f"The dataset directory does not exist: {dataset_root}")
        return

    # Get all projects
    projects = []
    for item in os.listdir(dataset_root):
        project_path = os.path.join(dataset_root, item)
        if os.path.isdir(project_path):
            projects.append((item, project_path))

    # Store all results
    all_results = {}
    project_metrics_summary = {}

    # Process each project
    for project_name, project_path in projects:
        print(f"start processing: {project_name}")

        try:
            # Process project
            project_data = predictor.process_project(project_path, project_name)

            if project_data["results"]:
                # Calculate project metrics
                metrics = calculator.calculate_metrics_for_project(project_data["results"])

                if metrics:
                    # Save project results
                    all_results[project_name] = project_data
                    project_metrics_summary[project_name] = metrics

                    # Print project metrics
                    print(f"    TP={metrics['TP']}, FP={metrics['FP']}, TN={metrics['TN']}, FN={metrics['FN']}")
                    print(f"    Precision={metrics['precision']}, Recall={metrics['recall']}, F1={metrics['f1_score']}")

            else:
                print(f"No file in project")

        except Exception as e:
            print(f"Fail to process project {project_name}")

    # Calculate overall metrics
    if all_results:
        overall_metrics = calculator.calculate_overall_metrics(all_results)

        # Print overall metrics
        print(f"    TP={overall_metrics['TP']}, FP={overall_metrics['FP']}")
        print(f"    TN={overall_metrics['TN']}, FN={overall_metrics['FN']}")
        print(f"    Precision: {overall_metrics['precision']}")
        print(f"    Recall:    {overall_metrics['recall']}")
        print(f"    F1-Score:  {overall_metrics['f1_score']}")
        print(f"    Accuracy:  {overall_metrics['accuracy']}")
        print(f"    AUC:       {overall_metrics['auc']}")

        # Save results to files
        save_results(all_results, project_metrics_summary, overall_metrics)
    else:
        print("\n No project processed successfully")

    end_time = time.time()
    total_time = end_time - start_time

if __name__ == "__main__":
    main()