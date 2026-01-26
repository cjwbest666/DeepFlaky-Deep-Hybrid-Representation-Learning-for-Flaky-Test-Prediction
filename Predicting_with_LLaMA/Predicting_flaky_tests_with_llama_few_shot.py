import os
import json
import time
import requests
from typing import List, Dict


class LlamaFlakyPredictor:
    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "llama3:8b"):
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api/generate"

        try:
            response = requests.get(f"{base_url}/api/tags")
            print("Successfully connected to Ollama")
        except Exception as e:
            print(f"Unable to connect to Ollama: {e}")
            print("Please ensure Ollama is running: ollama serve")
            exit(1)

        self.few_shot_examples = [
            {
                "code": """@Deployment(resources="org/activiti/engine/test/api/event/JobEventsTest.testJobEntityEvents.bpmn20.xml") 
public void testActivityTimeOutEvent(){
    ProcessInstance processInstance=runtimeService.startProcessInstanceByKey("testJobEvents");
    Job theJob=managementService.createTimerJobQuery().processInstanceId(processInstance.getId()).singleResult();
    assertNotNull(theJob);
    Calendar tomorrow=Calendar.getInstance();
    tomorrow.add(Calendar.DAY_OF_YEAR,1);
    processEngineConfiguration.getClock().setCurrentTime(tomorrow.getTime());
    waitForJobExecutorToProcessAllJobs(2000,1000);
    assertEquals(1,listener.getEventsReceived().size());
    ActivitiEvent activitiEvent=listener.getEventsReceived().get(0);
    assertEquals("ACTIVITY_CANCELLED event expected",ActivitiEventType.ACTIVITY_CANCELLED,activitiEvent.getType());
    ActivitiActivityCancelledEvent cancelledEvent=(ActivitiActivityCancelledEvent)activitiEvent;
    assertTrue("TIMER is the cause of the cancellation",cancelledEvent.getCause() instanceof JobEntity);
}""",
                "label": "Flaky",
                "reason": "Contains time-sensitive operations and asynchronous waits, depends on system clock and timer execution time"
            },
            {
                "code": """@Test public void readTest2() throws IOException {
    for (int k=MIN_LEN; k <= MAX_LEN; k+=DELTA) {
        for (    WriteType op : WriteType.values()) {
            int fileId=TestUtils.createByteFile(mTfs,"/root/testFile_" + k + "_"+ op,op,k);
            TachyonFile file=mTfs.getFile(fileId);
            InStream is=(k < MEAN ? file.getInStream(ReadType.CACHE) : file.getInStream(ReadType.NO_CACHE));
            if (k == 0) {
                Assert.assertTrue(is instanceof EmptyBlockInStream);
            }
            else {
                Assert.assertTrue(is instanceof BlockInStream);
            }
            byte[] ret=new byte[k];
            Assert.assertEquals(k,is.read(ret));
            Assert.assertTrue(TestUtils.equalIncreasingByteArray(k,ret));
            is.close();
            is=(k < MEAN ? file.getInStream(ReadType.CACHE) : file.getInStream(ReadType.NO_CACHE));
            if (k == 0) {
                Assert.assertTrue(is instanceof EmptyBlockInStream);
            }
            else {
                Assert.assertTrue(is instanceof BlockInStream);
            }
            ret=new byte[k];
            Assert.assertEquals(k,is.read(ret));
            Assert.assertTrue(TestUtils.equalIncreasingByteArray(k,ret));
            is.close();
        }
    }
}""",
                "label": "Flaky",
                "reason": "Involves file system I/O operations, potential resource competition and external dependencies"
            },
            {
                "code": """@Test public void convertXMLToElementShouldSetTheImplementationFromXMLImplementationAttribute() throws Exception {
    given(reader.getAttributeValue(null,BpmnXMLConstants.ATTRIBUTE_TASK_IMPLEMENTATION)).willReturn("myConnector");
    BaseElement element=converter.convertXMLToElement(reader,new BpmnModel());
    assertThat(((ServiceTask)element).getImplementation()).isEqualTo("myConnector");
}""",
                "label": "Non-Flaky",
                "reason": "Pure logic conversion, no time, concurrency or external resource dependencies"
            },
            {
                "code": """@Test public void computeBlockIdTest(){
    Assert.assertEquals(1073741824,BlockInfo.computeBlockId(1,0));
    Assert.assertEquals(1073741825,BlockInfo.computeBlockId(1,1));
    Assert.assertEquals(2147483646,BlockInfo.computeBlockId(1,1073741822));
    Assert.assertEquals(2147483647,BlockInfo.computeBlockId(1,1073741823));
    Assert.assertEquals(3221225472L,BlockInfo.computeBlockId(3,0));
    Assert.assertEquals(3221225473L,BlockInfo.computeBlockId(3,1));
    Assert.assertEquals(4294967294L,BlockInfo.computeBlockId(3,1073741822));
    Assert.assertEquals(4294967295L,BlockInfo.computeBlockId(3,1073741823));
}""",
                "label": "Non-Flaky",
                "reason": "Deterministic calculation, pure mathematical operations, no external dependencies"
            },
            {
                "code": """@Test public void doubleConversionValidation() throws Exception {
    BpmnModel bpmnModel=readJsonFile();
    bpmnModel=convertToJsonAndBack(bpmnModel);
    validateModel(bpmnModel);
}""",
                "label": "Non-Flaky",
                "reason": "This is a stable unit test with the following characteristics: 1. Deterministic, 2. No state dependencies, 3. No time sensitivity, 4. No concurrency, 5. Pure logic"
            }
        ]

        try:
            response = requests.get(f"{base_url}/api/tags")
        except Exception as e:
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
        examples_text = "Examples of flaky test and non-flaky test:\n\n"

        for i, example in enumerate(self.few_shot_examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"```java\n{example['code']}\n```\n"
            examples_text += f"Analysis: {example['reason']}\n"
            examples_text += f"Judgment: {example['label']}\n\n"

        prompt = f"""{examples_text}
        Based on the above examples, analyze whether the following Java test method is a flaky test.

        Flaky test characteristics:
        1. Concurrency issues (multi-threaded race conditions)
        2. Asynchronous operations and waits (sleep, wait, await)
        3. External resource dependencies (files, network, database)
        4. Time sensitivity (depends on system clock)
        5. Randomness (random number generation)
        6. Test order dependencies
        7. Resource leaks

        Non-flaky test characteristics:
        1. Pure logical operations
        2. No external resource dependencies
        3. Deterministic results
        4. No concurrent operations
        5. No time dependencies

        Output only "Flaky" or "Non-Flaky", without any other content.

        Test code to analyze:
        ```java
        {java_code}
        Judgment:"""
        return prompt

    def predict(self, java_code: str, max_retries: int = 3) -> str:
        prompt = self.construct_prompt(java_code)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 100
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "").strip()

                    response_text = response_text.replace('"', '').replace("'", "")

                    if "Flaky" in response_text:
                        return "Flaky"
                    elif "Non-Flaky" in response_text or "NonFlaky" in response_text:
                        return "Non-Flaky"
                    else:
                        flaky_keywords = ["concurrent", "async", "wait", "sleep", "timer", "time",
                                          "network", "file", "database", "external", "random"]
                        if any(keyword in response_text.lower() for keyword in flaky_keywords):
                            return "Flaky"
                        else:
                            return "Unknown"
            except Exception as e:
                print(f"Prediction failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(1)

        return "Error"

    def process_single_file(self, file_path: str, true_label: str) -> Dict:
        java_code = self.read_java_test_file(file_path)
        if not java_code:
            return None

        prediction = self.predict(java_code)
        time.sleep(0.3)

        return {
            "file_name": os.path.basename(file_path),
            "true_label": true_label,
            "predicted_label": prediction,
            "file_path": file_path
        }


class MetricsCalculator:
    """Metrics calculator"""

    @staticmethod
    def calculate_confusion_matrix(results: List[Dict]) -> Dict:
        """Calculate confusion matrix"""
        tp = fp = tn = fn = 0

        for result in results:
            true = result["true_label"]
            pred = result["predicted_label"]

            if true == "Flaky":
                if pred == "Flaky":
                    tp += 1
                else:
                    fn += 1
            else:
                if pred == "Flaky":
                    fp += 1
                else:
                    tn += 1

        return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

    @staticmethod
    def calculate_metrics(confusion: Dict) -> Dict:
        """Calculate performance metrics"""
        tp = confusion["TP"]
        fp = confusion["FP"]
        tn = confusion["TN"]
        fn = confusion["FN"]
        total = tp + fp + tn + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0

        return {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "total": total,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4)
        }


def process_all_projects():
    """Main function to process all projects"""
    print("Using Few-shot prompt (3 flaky, 2 non-flaky examples)")
    predictor = LlamaFlakyPredictor()
    calculator = MetricsCalculator()

    dataset_root = "dataset"
    if not os.path.exists(dataset_root):
        print(f"Dataset directory does not exist: {dataset_root}")
        return

    projects = []
    for item in os.listdir(dataset_root):
        project_path = os.path.join(dataset_root, item)
        if os.path.isdir(project_path):
            projects.append((item, project_path))

    print(f"Found {len(projects)} projects")

    all_project_results = {}
    all_project_metrics = {}

    processed_projects = 0
    for project_name, project_path in projects:
        print(f"Processing project: {project_name}")

        project_results = []

        flaky_dir = os.path.join(project_path, "flakyMethods")
        flaky_count = 0
        if os.path.exists(flaky_dir):
            import glob
            flaky_files = glob.glob(os.path.join(flaky_dir, "*.java"))
            for file_path in flaky_files:
                result = predictor.process_single_file(file_path, "Flaky")
                if result:
                    project_results.append(result)
                    flaky_count += 1

                if flaky_count >= 20:
                    break

        nonflaky_dir = os.path.join(project_path, "nonFlakyMethods")
        nonflaky_count = 0
        if os.path.exists(nonflaky_dir):
            nonflaky_files = glob.glob(os.path.join(nonflaky_dir, "*.java"))
            for file_path in nonflaky_files:
                result = predictor.process_single_file(file_path, "Non-Flaky")
                if result:
                    project_results.append(result)
                    nonflaky_count += 1

                if nonflaky_count >= flaky_count * 3:
                    break

        if project_results:
            confusion = calculator.calculate_confusion_matrix(project_results)
            metrics = calculator.calculate_metrics(confusion)

            all_project_results[project_name] = project_results
            all_project_metrics[project_name] = metrics

            print(f"  Processed: {flaky_count} flaky, {nonflaky_count} non-flaky")
            print(f"    Metrics: TP={metrics['TP']}, FP={metrics['FP']}, TN={metrics['TN']}, FN={metrics['FN']}")
            print(f"             Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")

            processed_projects += 1
        else:
            print(f"  No files processed")

        if processed_projects >= 3:
            print(f"Reached test project limit (3)")
            break

    if all_project_results:
        print("Calculating overall metrics...")

        all_results = []
        for project_name, results in all_project_results.items():
            all_results.extend(results)

        overall_confusion = calculator.calculate_confusion_matrix(all_results)
        overall_metrics = calculator.calculate_metrics(overall_confusion)

        print("Overall Metrics:")
        print(f"  Total projects: {len(all_project_metrics)}")
        print(f"  Total samples: {overall_metrics['total']}")
        print(f"  Confusion matrix:")
        print(f"    TP={overall_metrics['TP']} (correctly predicted flaky)")
        print(f"    FP={overall_metrics['FP']} (false positives)")
        print(f"    TN={overall_metrics['TN']} (correctly predicted non-flaky)")
        print(f"    FN={overall_metrics['FN']} (false negatives)")
        print(f"  Performance metrics:")
        print(f"    Precision: {overall_metrics['precision']:.4f}")
        print(f"    Recall:    {overall_metrics['recall']:.4f}")
        print(f"    F1-Score:  {overall_metrics['f1_score']:.4f}")
        print(f"    Accuracy:  {overall_metrics['accuracy']:.4f}")

        save_results(all_project_metrics, overall_metrics)
    else:
        print("No projects successfully processed")


def save_results(project_metrics: Dict, overall_metrics: Dict):
    """Save results"""
    suffix = "_fewshot"
    output_dir = f"llama_results{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    import csv
    csv_path = os.path.join(output_dir, "project_metrics.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Project", "TP", "FP", "TN", "FN", "Precision", "Recall", "F1-Score", "Accuracy", "Samples"])

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
                metrics["total"]
            ])

    summary_path = os.path.join(output_dir, "overall_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("LLAMA FLAKY TEST PREDICTION - FEW-SHOT\n")
        f.write(f"Total Projects: {len(project_metrics)}\n")
        f.write(f"Total Samples: {overall_metrics['total']}\n\n")

        f.write("CONFUSION MATRIX:\n")
        f.write(f"True Positives (TP): {overall_metrics['TP']}\n")
        f.write(f"False Positives (FP): {overall_metrics['FP']}\n")
        f.write(f"True Negatives (TN): {overall_metrics['TN']}\n")
        f.write(f"False Negatives (FN): {overall_metrics['FN']}\n\n")

        f.write("PERFORMANCE METRICS:\n")
        f.write(f"Precision: {overall_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {overall_metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {overall_metrics['f1_score']:.4f}\n")
        f.write(f"Accuracy:  {overall_metrics['accuracy']:.4f}\n")

    print(f"Results saved to directory: {output_dir}")
    print(f"  - Project metrics: {csv_path}")
    print(f"  - Overall summary: {summary_path}")


def test_few_shot_examples():
    """Test few-shot examples themselves"""
    print("Testing Few-shot examples")
    predictor = LlamaFlakyPredictor()

    print("Testing Few-shot examples predictions:")
    for i, example in enumerate(predictor.few_shot_examples, 1):
        print(f"Example {i} (should be {example['label']}):")
        print(f"Code snippet: {example['code'][:100]}...")

        prediction = predictor.predict(example['code'])
        print(f"Model prediction: {prediction}")
        print(f"Correct: {prediction == example['label']}")


if __name__ == "__main__":
    print("LLAMA FLAKY TEST PREDICTION WITH FEW-SHOT LEARNING")

    print(f"Using {len(LlamaFlakyPredictor().few_shot_examples)} Few-shot examples:")
    for i, example in enumerate(LlamaFlakyPredictor().few_shot_examples, 1):
        print(f"  {i}. {example['label']}: {example['reason']}")

    process_all_projects()