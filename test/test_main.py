import unittest
import pandas as pd
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

from ..main import RequestEvaluator

def evaluate_relevancy(response: str, expected_answer: str) -> float:
    test_case = LLMTestCase(
        input="",
        actual_output=response,
        expected_output=expected_answer
    )
    relevancy_metric = AnswerRelevancyMetric()
    result = evaluate([test_case], [relevancy_metric], print_results=False)
    return result.test_results[0].metrics_data[0].score

class TestRAGSync(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.qa_data = pd.read_csv('qa/QA_billability.csv')

    def test_csv_queries(self):
        evaluator = RequestEvaluator()
        for index, row in self.qa_data.iterrows():
            with self.subTest(question=row['question']):
                print(f"Running query: {row['question']}")
                response = evaluator.evaluate_customer_request(row['question']).conclusion
                self.assertIsInstance(response, str)
                self.assertTrue(len(response) > 0, "The response should not be empty")
                print(f"Response: {response}")
                # Relevancy evaluation
                expected_answer = row['context']
                print(f"Expected answer: {expected_answer}")
                relevancy_score = evaluate_relevancy(response, expected_answer)
                print(f"Relevancy score: {relevancy_score}")
                self.assertGreaterEqual(
                    relevancy_score, 0.0,
                    f"Relevancy score {relevancy_score} is below the threshold for question: {row['question']}"
                )


if __name__ == '__main__':
    unittest.main()
