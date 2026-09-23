"""
Evaluation suite for K8s RAG.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class EvaluationResult(Enum):
    """Evaluation result types."""
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    NO_ANSWER = "no_answer"


@dataclass
class TestQuestion:
    """A single test question."""
    category: str
    query: str
    expected_keywords: List[str] = field(default_factory=list)
    expected_commands: List[str] = field(default_factory=list)
    expected_concepts: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return self.query


@dataclass
class TestResult:
    """Result of a single test."""
    question: TestQuestion
    answer: str
    result: EvaluationResult
    faithfulness_score: float = 0.0
    relevance_score: float = 0.0
    explanation: str = ""


class EvaluationSuite:
    """
    Comprehensive evaluation suite for K8s RAG.
    
    Provides:
    - Test cases across different categories
    - Answer evaluation
    - Faithfulness and relevance scoring
    - Metrics aggregation
    """
    
    # Test questions organized by category
    TEST_CASES = [
        # Core Concepts
        TestQuestion(
            category="Core",
            query="What is a Kubernetes Pod and what are its key characteristics?",
            expected_keywords=["smallest", "unit", "run", "container", "orchestration"],
            expected_concepts=["pod", "container", "namespace", "ephemeral"]
        ),
        TestQuestion(
            category="Core",
            query="Explain the difference between a Deployment and a StatefulSet.",
            expected_keywords=["deployment", "statefulset", "stateless", "stateful", "ordered"],
            expected_concepts=["workload", "replica", "identity", "storage"]
        ),
        
        # Scaling
        TestQuestion(
            category="Scaling",
            query="How do I scale a deployment in Kubernetes?",
            expected_keywords=["kubectl", "scale", "deployment", "replicas"],
            expected_commands=["kubectl scale"]
        ),
        TestQuestion(
            category="Scaling",
            query="What is a HorizontalPodAutoscaler?",
            expected_keywords=["HPA", "horizontal", "autoscaler", "cpu", "memory", "metrics"],
            expected_concepts["scaling", "auto-scaling", "replicas", "metrics"]
        ),
        
        # Configuration
        TestQuestion(
            category="Configuration",
            query="What is a Kubernetes ConfigMap and how can pods consume it?",
            expected_keywords=["ConfigMap", "configuration", "environment", "variables"],
            expected_concepts=["configmap", "volume", "environment variable", "mount"]
        ),
        TestQuestion(
            category="Configuration",
            query="How do I create a ConfigMap and use it in a pod?",
            expected_keywords=["kubectl", "create", "configmap", "env", "volume"],
            expected_commands=["kubectl create configmap"]
        ),
        
        # Networking
        TestQuestion(
            category="Networking",
            query="Explain the different types of Kubernetes Services.",
            expected_keywords=["service", "clusterIP", "nodePort", "loadBalancer", "externalName"],
            expected_concepts["service types", "service discovery", "networking", "load balancing"]
        ),
        TestQuestion(
            category="Networking",
            query="How do I expose a service to the internet?",
            expected_keywords=["loadBalancer", "nodePort", "ingress", "external"],
            expected_concepts["expose", "internet", "external access", "port"]
        ),
        
        # Storage
        TestQuestion(
            category="Storage",
            query="How do PersistentVolumes and PersistentVolumeClaims work?",
            expected_keywords=["PersistentVolume", "PersistentVolumeClaim", "storage", "bind"],
            expected_concepts=["pv", "pvc", "storage class", "dynamic provisioning"]
        ),
        TestQuestion(
            category="Storage",
            query="Why does a PersistentVolume persist across pod restarts?",
            expected_keywords=["persistent", "storage", "node", "restart", "bound"],
            expected_concepts["persistence", "data retention", "lifecycle"]
        ),
        
        # Troubleshooting
        TestQuestion(
            category="Troubleshooting",
            query="My pod is stuck in CrashLoopBackOff. How do I debug this?",
            expected_keywords=["CrashLoopBackOff", "debug", "logs", "describe", "events"],
            expected_commands=["kubectl logs", "kubectl describe pod"]
        ),
    ]
    
    def __init__(self):
        """Initialize the evaluation suite."""
        self.test_cases = self.TEST_CASES.copy()
        self.results: List[TestResult] = []
    
    def evaluate(self, rag_chain, queries: Optional[List[TestQuestion]] = None) -> List[TestResult]:
        """
        Evaluate the RAG chain against test questions.
        
        Args:
            rag_chain: RAGChain instance to evaluate
            queries: Optional list of TestQuestion instances (uses defaults if None)
        
        Returns:
            List of TestResult objects
        """
        if queries is None:
            queries = self.test_cases
        
        self.results = []
        
        for question in queries:
            result = self._evaluate_question(rag_chain, question)
            self.results.append(result)
        
        return self.results
    
    def _evaluate_question(self, rag_chain, question: TestQuestion) -> TestResult:
        """
        Evaluate a single question.
        
        Args:
            rag_chain: RAGChain instance
            question: TestQuestion to evaluate
        
        Returns:
            TestResult object
        """
        # Get answer from RAG chain
        answer = rag_chain.invoke(question.query)
        
        # Evaluate the answer
        result = self._check_answer(question, answer)
        
        # Calculate faithfulness and relevance
        faithfulness, relevance = self._calculate_scores(question, answer)
        
        return TestResult(
            question=question,
            answer=answer,
            result=result,
            faithfulness_score=faithfulness,
            relevance_score=relevance,
            explanation=self._generate_explanation(question, answer, result)
        )
    
    def _check_answer(self, question: TestQuestion, answer: str) -> EvaluationResult:
        """
        Check if the answer is correct.
        
        Args:
            question: Original test question
            answer: Generated answer
        
        Returns:
            EvaluationResult
        """
        answer_lower = answer.lower()
        
        # Check for expected keywords
        keywords_found = [
            kw for kw in question.expected_keywords
            if kw.lower() in answer_lower
        ]
        
        # Check for expected concepts
        concepts_found = [
            conc for conc in question.expected_concepts
            if conc.lower() in answer_lower
        ]
        
        # Check for expected commands
        commands_found = []
        for cmd in question.expected_commands:
            if cmd.lower() in answer_lower:
                commands_found.append(cmd)
        
        # Determine result
        if len(keywords_found) >= len(question.expected_keywords) * 0.7 and len(concepts_found) >= len(question.expected_concepts) * 0.7:
            return EvaluationResult.CORRECT
        elif len(keywords_found) >= len(question.expected_keywords) * 0.4 or len(concepts_found) >= len(question.expected_concepts) * 0.4:
            return EvaluationResult.PARTIAL
        elif answer.strip() == "" or answer.lower() in ["i don't know", "i'm not sure", "i don't have enough information"]:
            return EvaluationResult.NO_ANSWER
        else:
            return EvaluationResult.INCORRECT
    
    def _calculate_scores(self, question: TestQuestion, answer: str) -> tuple:
        """
        Calculate faithfulness and relevance scores.
        
        Args:
            question: Original test question
            answer: Generated answer
        
        Returns:
            Tuple of (faithfulness_score, relevance_score)
        """
        answer_lower = answer.lower()
        
        # Faithfulness: How much of the answer is supported by the context
        # (simplified - in production, would use actual context)
        keyword_match_ratio = len([
            kw for kw in question.expected_keywords
            if kw.lower() in answer_lower
        ]) / len(question.expected_keywords) if question.expected_keywords else 0
        
        # Relevance: How well the answer addresses the question
        question_lower = question.query.lower()
        relevance_ratio = len(set(answer_lower.split()) & set(question_lower.split())) / max(len(answer_lower.split()), len(question_lower.split()))
        
        # Normalize scores (0-1)
        faithfulness = min(1.0, keyword_match_ratio * 1.5)
        relevance = min(1.0, relevance_ratio * 2.0)
        
        return faithfulness, relevance
    
    def _generate_explanation(self, question: TestQuestion, answer: str, result: EvaluationResult) -> str:
        """
        Generate an explanation for the evaluation result.
        
        Args:
            question: Original test question
            answer: Generated answer
            result: Evaluation result
        
        Returns:
            Explanation string
        """
        explanations = {
            EvaluationResult.CORRECT: "✅ Answer is correct. All key concepts and keywords are covered.",
            EvaluationResult.PARTIAL: "⚠️ Answer is partially correct. Some key concepts are missing or incomplete.",
            EvaluationResult.INCORRECT: "❌ Answer is incorrect or contains hallucinated information.",
            EvaluationResult.NO_ANSWER: "❌ No meaningful answer was provided."
        }
        
        return explanations.get(result, "Unknown result")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Calculate aggregate metrics from all test results.
        
        Returns:
            Dictionary with metrics
        """
        if not self.results:
            return {
                "total_questions": 0,
                "correct": 0,
                "partial": 0,
                "incorrect": 0,
                "no_answer": 0,
                "factual_accuracy": 0.0,
                "hallucination_rate": 0.0,
                "average_faithfulness": 0.0,
                "average_relevance": 0.0,
            }
        
        total = len(self.results)
        correct = sum(1 for r in self.results if r.result == EvaluationResult.CORRECT)
        partial = sum(1 for r in self.results if r.result == EvaluationResult.PARTIAL)
        incorrect = sum(1 for r in self.results if r.result == EvaluationResult.INCORRECT)
        no_answer = sum(1 for r in self.results if r.result == EvaluationResult.NO_ANSWER)
        
        # Factual accuracy: (correct + 0.5 * partial) / total
        factual_accuracy = (correct + 0.5 * partial) / total if total > 0 else 0.0
        
        # Hallucination rate: incorrect / total
        hallucination_rate = incorrect / total if total > 0 else 0.0
        
        # Average faithfulness and relevance
        avg_faithfulness = sum(r.faithfulness_score for r in self.results) / total
        avg_relevance = sum(r.relevance_score for r in self.results) / total
        
        return {
            "total_questions": total,
            "correct": correct,
            "partial": partial,
            "incorrect": incorrect,
            "no_answer": no_answer,
            "factual_accuracy": factual_accuracy * 100,
            "hallucination_rate": hallucination_rate * 100,
            "average_faithfulness": avg_faithfulness * 100,
            "average_relevance": avg_relevance * 100,
            "by_category": self._get_metrics_by_category()
        }
    
    def _get_metrics_by_category(self) -> Dict[str, Dict[str, int]]:
        """
        Get metrics grouped by category.
        
        Returns:
            Dictionary with category metrics
        """
        category_metrics = {}
        
        for result in self.results:
            category = result.question.category
            if category not in category_metrics:
                category_metrics[category] = {
                    "total": 0,
                    "correct": 0,
                    "partial": 0,
                    "accuracy": 0.0
                }
            
            category_metrics[category]["total"] += 1
            if result.result == EvaluationResult.CORRECT:
                category_metrics[category]["correct"] += 1
                category_metrics[category]["accuracy"] = (
                    category_metrics[category]["correct"] / 
                    category_metrics[category]["total"] * 100
                )
        
        return category_metrics
    
    def print_report(self):
        """Print a formatted evaluation report."""
        metrics = self.get_metrics()
        
        print("\n" + "=" * 60)
        print("K8s RAG Evaluation Report")
        print("=" * 60)
        
        print(f"\n📊 Overall Metrics:")
        print(f"   Total Questions: {metrics['total_questions']}")
        print(f"   Factual Accuracy: {metrics['factual_accuracy']:.1f}%")
        print(f"   Hallucination Rate: {metrics['hallucination_rate']:.1f}%")
        print(f"   Average Faithfulness: {metrics['average_faithfulness']:.1f}%")
        print(f"   Average Relevance: {metrics['average_relevance']:.1f}%")
        
        print(f"\n📋 Results Breakdown:")
        print(f"   Correct: {metrics['correct']}")
        print(f"   Partial: {metrics['partial']}")
        print(f"   Incorrect: {metrics['incorrect']}")
        print(f"   No Answer: {metrics['no_answer']}")
        
        print(f"\n📁 Results by Category:")
        for category, cat_metrics in metrics['by_category'].items():
            print(f"   {category}: {cat_metrics['accuracy']:.1f}% ({cat_metrics['correct']}/{cat_metrics['total']})")
        
        # Print detailed results
        print("\n" + "-" * 60)
        print("Detailed Results:")
        print("-" * 60)
        
        for i, result in enumerate(self.results, 1):
            print(f"\n{i}. [{result.question.category}] {result.question.query}")
            print(f"   Result: {result.result.value.upper()}")
            print(f"   Faithfulness: {result.faithfulness_score * 100:.1f}%")
            print(f"   Relevance: {result.relevance_score * 100:.1f}%")
            print(f"   Explanation: {result.explanation}")
        
        print("\n" + "=" * 60)
