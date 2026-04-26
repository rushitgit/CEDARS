# Model Training Ideation for CEDARS

This document outlines potential model training approaches and code architectures for enhancing the CEDARS (Clinical Event Detection and Recording System) NLP capabilities.

## Current Architecture Overview

CEDARS currently uses:
- **spaCy** with `en_core_sci_lg` (scientific/biomedical model) for NLP processing
- **Pattern-based matching** via spaCy's `Matcher` for keyword/query detection
- **Negation detection** using dependency parsing
- **PINES** integration for ML-based predictions via external API

## Proposed Training Code Architectures

### 1. Fine-tuned Clinical Event Classifier

Train a transformer-based classifier to identify clinical events from medical text.

```python
# cedars/training/event_classifier.py
"""
Clinical Event Classification Model Training

This module provides training utilities for a transformer-based
classifier that identifies clinical events in medical notes.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    output_dir: str = "./models/event_classifier"
    early_stopping_patience: int = 3
    

class ClinicalEventDataset(Dataset):
    """Dataset for clinical event classification."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class ClinicalEventTrainer:
    """Trainer for clinical event classification models."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = None
        
    def prepare_data(
        self,
        annotations: List[Dict],
        test_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[Dataset, Dataset]:
        """
        Prepare training and validation datasets from CEDARS annotations.
        
        Args:
            annotations: List of annotation dictionaries from MongoDB
            test_size: Proportion of data for validation
            stratify: Whether to stratify split by label
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        texts = []
        labels = []
        
        for annot in annotations:
            sentence = annot.get("sentence", "")
            is_event = 1 if annot.get("reviewed") == "EVENT_CONFIRMED" else 0
            texts.append(sentence)
            labels.append(is_event)
        
        stratify_labels = labels if stratify else None
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=42
        )
        
        train_dataset = ClinicalEventDataset(
            train_texts, train_labels,
            self.tokenizer, self.config.max_length
        )
        val_dataset = ClinicalEventDataset(
            val_texts, val_labels,
            self.tokenizer, self.config.max_length
        )
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        f1 = f1_score(labels, predictions, average="binary")
        report = classification_report(labels, predictions, output_dict=True)
        
        return {
            "f1": f1,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
        }
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        class_weights: Optional[List[float]] = None
    ):
        """
        Train the clinical event classifier.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            class_weights: Optional weights for imbalanced classes
        """
        num_labels = 2
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=num_labels
        )
        
        if class_weights:
            self.model.config.class_weights = class_weights
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=100,
            fp16=torch.cuda.is_available(),
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                )
            ]
        )
        
        trainer.train()
        
        return trainer
    
    def save_model(self, path: str):
        """Save the trained model."""
        if self.model:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
    
    def load_model(self, path: str):
        """Load a trained model."""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)


def train_from_cedars_db(mongo_uri: str, db_name: str, config: TrainingConfig):
    """
    Train a model using annotations from CEDARS MongoDB.
    
    Example usage:
        config = TrainingConfig(
            model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            num_epochs=5,
            batch_size=8
        )
        train_from_cedars_db("mongodb://localhost:27017", "cedars", config)
    """
    from pymongo import MongoClient
    
    client = MongoClient(mongo_uri)
    db = client[db_name]
    
    annotations = list(db["ANNOTATIONS"].find({
        "reviewed": {"$in": ["EVENT_CONFIRMED", "REVIEWED"]}
    }))
    
    if len(annotations) < 100:
        raise ValueError(
            f"Insufficient training data: {len(annotations)} annotations. "
            "Minimum 100 required for meaningful training."
        )
    
    trainer = ClinicalEventTrainer(config)
    train_dataset, val_dataset = trainer.prepare_data(annotations)
    
    positive_ratio = sum(1 for a in annotations if a.get("reviewed") == "EVENT_CONFIRMED") / len(annotations)
    class_weights = [1.0, 1.0 / positive_ratio] if positive_ratio < 0.3 else None
    
    trainer.train(train_dataset, val_dataset, class_weights)
    trainer.save_model(config.output_dir)
    
    return trainer
```

### 2. Named Entity Recognition (NER) for Medical Concepts

Custom NER model for extracting medical entities relevant to clinical events.

```python
# cedars/training/medical_ner.py
"""
Medical Named Entity Recognition Training

Train a custom NER model for extracting medical concepts
such as symptoms, diagnoses, procedures, and medications.
"""

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass, field


@dataclass
class NERTrainingConfig:
    """Configuration for NER model training."""
    base_model: str = "en_core_sci_lg"
    output_dir: str = "./models/medical_ner"
    n_iter: int = 30
    batch_size: int = 8
    dropout: float = 0.35
    learn_rate: float = 0.001
    entity_labels: List[str] = field(default_factory=lambda: [
        "SYMPTOM",
        "DIAGNOSIS",
        "PROCEDURE",
        "MEDICATION",
        "ANATOMY",
        "LAB_VALUE",
        "CLINICAL_EVENT"
    ])


class MedicalNERTrainer:
    """Trainer for medical NER models."""
    
    def __init__(self, config: NERTrainingConfig):
        self.config = config
        self.nlp = None
        
    def create_training_data(
        self,
        annotations: List[Dict],
        validation_split: float = 0.2
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Convert CEDARS annotations to spaCy NER training format.
        
        Expected annotation format:
        {
            "sentence": "Patient presents with chest pain",
            "entities": [
                {"start": 22, "end": 32, "label": "SYMPTOM"}
            ]
        }
        """
        training_data = []
        
        for annot in annotations:
            text = annot.get("sentence", "")
            entities = annot.get("entities", [])
            
            ents = [(e["start"], e["end"], e["label"]) for e in entities]
            training_data.append((text, {"entities": ents}))
        
        random.shuffle(training_data)
        split_idx = int(len(training_data) * (1 - validation_split))
        
        return training_data[:split_idx], training_data[split_idx:]
    
    def initialize_model(self):
        """Initialize the spaCy model with NER component."""
        self.nlp = spacy.load(self.config.base_model)
        
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")
        
        for label in self.config.entity_labels:
            ner.add_label(label)
            
        return self.nlp
    
    def train(
        self,
        train_data: List[Tuple],
        val_data: List[Tuple]
    ) -> Dict[str, float]:
        """
        Train the NER model.
        
        Returns:
            Dictionary of final evaluation metrics
        """
        if self.nlp is None:
            self.initialize_model()
        
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.resume_training()
            
            for iteration in range(self.config.n_iter):
                random.shuffle(train_data)
                losses = {}
                
                batches = minibatch(
                    train_data,
                    size=compounding(4.0, self.config.batch_size, 1.001)
                )
                
                for batch in batches:
                    examples = []
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)
                    
                    self.nlp.update(
                        examples,
                        drop=self.config.dropout,
                        losses=losses
                    )
                
                val_scores = self.evaluate(val_data)
                print(f"Iteration {iteration + 1}: Loss={losses.get('ner', 0):.4f}, "
                      f"F1={val_scores['f1']:.4f}")
        
        return self.evaluate(val_data)
    
    def evaluate(self, test_data: List[Tuple]) -> Dict[str, float]:
        """Evaluate the NER model."""
        scorer = {"tp": 0, "fp": 0, "fn": 0}
        
        for text, annotations in test_data:
            doc = self.nlp(text)
            pred_ents = set((ent.start_char, ent.end_char, ent.label_) for ent in doc.ents)
            gold_ents = set((e[0], e[1], e[2]) for e in annotations.get("entities", []))
            
            scorer["tp"] += len(pred_ents & gold_ents)
            scorer["fp"] += len(pred_ents - gold_ents)
            scorer["fn"] += len(gold_ents - pred_ents)
        
        precision = scorer["tp"] / (scorer["tp"] + scorer["fp"]) if (scorer["tp"] + scorer["fp"]) > 0 else 0
        recall = scorer["tp"] / (scorer["tp"] + scorer["fn"]) if (scorer["tp"] + scorer["fn"]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def save_model(self, path: Optional[str] = None):
        """Save the trained model."""
        output_path = Path(path or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(output_path)
    
    def load_model(self, path: str):
        """Load a trained model."""
        self.nlp = spacy.load(path)
```

### 3. Active Learning Pipeline

Implement active learning to efficiently use annotator time.

```python
# cedars/training/active_learning.py
"""
Active Learning Pipeline for CEDARS

This module implements an active learning workflow that prioritizes
uncertain samples for human annotation, maximizing model improvement
with minimal labeling effort.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score
from scipy.stats import entropy
import torch
from torch.nn.functional import softmax
from abc import ABC, abstractmethod


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning."""
    initial_labeled_size: int = 100
    query_size: int = 20
    max_iterations: int = 50
    uncertainty_threshold: float = 0.4
    diversity_weight: float = 0.3
    

class UncertaintySampler(ABC):
    """Base class for uncertainty sampling strategies."""
    
    @abstractmethod
    def score(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate uncertainty scores for predictions."""
        pass


class EntropySampler(UncertaintySampler):
    """Entropy-based uncertainty sampling."""
    
    def score(self, predictions: np.ndarray) -> np.ndarray:
        return entropy(predictions, axis=1)


class MarginSampler(UncertaintySampler):
    """Margin-based uncertainty sampling (difference between top 2 predictions)."""
    
    def score(self, predictions: np.ndarray) -> np.ndarray:
        sorted_probs = np.sort(predictions, axis=1)
        return 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])


class LeastConfidenceSampler(UncertaintySampler):
    """Least confidence sampling."""
    
    def score(self, predictions: np.ndarray) -> np.ndarray:
        return 1 - np.max(predictions, axis=1)


class ActiveLearningPipeline:
    """Active learning pipeline for clinical event detection."""
    
    def __init__(
        self,
        model,
        config: ActiveLearningConfig,
        sampler: UncertaintySampler = None
    ):
        self.model = model
        self.config = config
        self.sampler = sampler or EntropySampler()
        self.labeled_indices = set()
        self.iteration_metrics = []
        
    def initialize_labeled_pool(
        self,
        X: np.ndarray,
        y: np.ndarray,
        strategy: str = "random"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the labeled pool with a small set of samples.
        
        Args:
            X: Feature matrix
            y: Labels
            strategy: "random" or "stratified"
            
        Returns:
            Initial labeled data (X_labeled, y_labeled)
        """
        n_samples = len(X)
        
        if strategy == "stratified":
            positive_idx = np.where(y == 1)[0]
            negative_idx = np.where(y == 0)[0]
            
            n_pos = min(
                len(positive_idx),
                self.config.initial_labeled_size // 2
            )
            n_neg = self.config.initial_labeled_size - n_pos
            
            selected_pos = np.random.choice(positive_idx, n_pos, replace=False)
            selected_neg = np.random.choice(negative_idx, n_neg, replace=False)
            selected = np.concatenate([selected_pos, selected_neg])
        else:
            selected = np.random.choice(
                n_samples,
                self.config.initial_labeled_size,
                replace=False
            )
        
        self.labeled_indices = set(selected)
        return X[selected], y[selected]
    
    def query(
        self,
        X_unlabeled: np.ndarray,
        unlabeled_indices: np.ndarray
    ) -> np.ndarray:
        """
        Select samples to query for labeling.
        
        Args:
            X_unlabeled: Unlabeled feature matrix
            unlabeled_indices: Original indices of unlabeled samples
            
        Returns:
            Indices of samples to query
        """
        if len(X_unlabeled) == 0:
            return np.array([])
        
        predictions = self.model.predict_proba(X_unlabeled)
        uncertainty_scores = self.sampler.score(predictions)
        
        if self.config.diversity_weight > 0:
            diversity_scores = self._compute_diversity(X_unlabeled)
            combined_scores = (
                (1 - self.config.diversity_weight) * uncertainty_scores +
                self.config.diversity_weight * diversity_scores
            )
        else:
            combined_scores = uncertainty_scores
        
        n_query = min(self.config.query_size, len(X_unlabeled))
        query_local_indices = np.argsort(combined_scores)[-n_query:]
        
        return unlabeled_indices[query_local_indices]
    
    def _compute_diversity(self, X: np.ndarray) -> np.ndarray:
        """
        Compute diversity scores based on distance to labeled samples.
        """
        if not hasattr(self, '_labeled_embeddings') or self._labeled_embeddings is None:
            return np.ones(len(X))
        
        distances = np.zeros(len(X))
        for i, x in enumerate(X):
            min_dist = np.min(np.linalg.norm(self._labeled_embeddings - x, axis=1))
            distances[i] = min_dist
        
        if np.max(distances) > 0:
            distances = distances / np.max(distances)
        
        return distances
    
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        oracle: Callable[[np.ndarray], np.ndarray],
        eval_X: Optional[np.ndarray] = None,
        eval_y: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Run the active learning loop.
        
        Args:
            X: Full feature matrix
            y: Full labels (for oracle simulation)
            oracle: Function that returns labels for given indices
            eval_X: Optional evaluation set features
            eval_y: Optional evaluation set labels
            
        Returns:
            List of metrics at each iteration
        """
        X_labeled, y_labeled = self.initialize_labeled_pool(X, y)
        
        for iteration in range(self.config.max_iterations):
            self.model.fit(X_labeled, y_labeled)
            
            if eval_X is not None and eval_y is not None:
                score = self.model.score(eval_X, eval_y)
            else:
                score = cross_val_score(
                    self.model, X_labeled, y_labeled, cv=3
                ).mean()
            
            metrics = {
                "iteration": iteration,
                "n_labeled": len(y_labeled),
                "score": score
            }
            self.iteration_metrics.append(metrics)
            
            unlabeled_mask = np.array([
                i not in self.labeled_indices for i in range(len(X))
            ])
            unlabeled_indices = np.where(unlabeled_mask)[0]
            
            if len(unlabeled_indices) == 0:
                print("All samples labeled. Stopping.")
                break
            
            query_indices = self.query(
                X[unlabeled_indices],
                unlabeled_indices
            )
            
            new_labels = oracle(query_indices)
            
            self.labeled_indices.update(query_indices)
            X_labeled = np.vstack([X_labeled, X[query_indices]])
            y_labeled = np.concatenate([y_labeled, new_labels])
            
            print(f"Iteration {iteration + 1}: "
                  f"Labeled={len(y_labeled)}, Score={score:.4f}")
        
        return self.iteration_metrics


class CEDARSActiveLearning:
    """
    Integration of active learning with CEDARS workflow.
    
    This class connects the active learning pipeline with
    CEDARS' annotation interface and database.
    """
    
    def __init__(self, db_client, model, config: ActiveLearningConfig):
        self.db = db_client
        self.pipeline = ActiveLearningPipeline(model, config)
        
    def get_priority_annotations(self, patient_id: str) -> List[Dict]:
        """
        Get prioritized list of sentences for annotation.
        
        Returns sentences ordered by model uncertainty,
        so annotators focus on most informative samples.
        """
        notes = list(self.db["NOTES"].find({"patient_id": patient_id}))
        
        sentences = []
        for note in notes:
            doc = self.pipeline.model.nlp(note["text"])
            for sent in doc.sents:
                sentences.append({
                    "note_id": note["text_id"],
                    "text": sent.text,
                    "start": sent.start_char,
                    "end": sent.end_char
                })
        
        if not sentences:
            return []
        
        texts = [s["text"] for s in sentences]
        predictions = self.pipeline.model.predict_proba(texts)
        uncertainty_scores = self.pipeline.sampler.score(predictions)
        
        for i, score in enumerate(uncertainty_scores):
            sentences[i]["uncertainty"] = float(score)
            sentences[i]["predicted_prob"] = float(predictions[i][1])
        
        sorted_sentences = sorted(
            sentences,
            key=lambda x: x["uncertainty"],
            reverse=True
        )
        
        return sorted_sentences
```

### 4. Multi-Task Learning for Clinical NLP

Joint training for multiple clinical NLP tasks.

```python
# cedars/training/multitask.py
"""
Multi-Task Learning for Clinical NLP

Train a single model for multiple clinical NLP tasks:
- Event Detection
- Negation Detection  
- Temporal Relation Extraction
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task model."""
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    hidden_size: int = 768
    num_event_classes: int = 2
    num_negation_classes: int = 2
    num_temporal_classes: int = 5
    dropout: float = 0.1
    task_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                "event": 1.0,
                "negation": 0.5,
                "temporal": 0.3
            }


class TaskHead(nn.Module):
    """Classification head for a specific task."""
    
    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class ClinicalMultiTaskModel(nn.Module):
    """
    Multi-task model for clinical NLP.
    
    Shared encoder with task-specific heads for:
    - Clinical event detection
    - Negation detection
    - Temporal relation classification
    """
    
    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.config = config
        
        self.encoder = AutoModel.from_pretrained(config.model_name)
        
        self.event_head = TaskHead(
            config.hidden_size,
            config.num_event_classes,
            config.dropout
        )
        self.negation_head = TaskHead(
            config.hidden_size,
            config.num_negation_classes,
            config.dropout
        )
        self.temporal_head = TaskHead(
            config.hidden_size,
            config.num_temporal_classes,
            config.dropout
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str = "all"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            task: Which task(s) to compute ("event", "negation", "temporal", "all")
            
        Returns:
            Dictionary of task outputs
        """
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = encoder_output.last_hidden_state[:, 0, :]
        
        outputs = {}
        
        if task in ["event", "all"]:
            outputs["event"] = self.event_head(pooled_output)
        
        if task in ["negation", "all"]:
            outputs["negation"] = self.negation_head(pooled_output)
            
        if task in ["temporal", "all"]:
            outputs["temporal"] = self.temporal_head(pooled_output)
        
        return outputs


class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss."""
    
    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.weights = config.task_weights
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted multi-task loss.
        
        Returns:
            Tuple of (total_loss, dict of individual losses)
        """
        losses = {}
        total_loss = 0.0
        
        for task_name, task_output in outputs.items():
            if task_name in labels and labels[task_name] is not None:
                task_loss = self.ce_loss(task_output, labels[task_name])
                losses[task_name] = task_loss
                total_loss += self.weights.get(task_name, 1.0) * task_loss
        
        return total_loss, losses


class MultiTaskTrainer:
    """Trainer for multi-task clinical model."""
    
    def __init__(
        self,
        model: ClinicalMultiTaskModel,
        config: MultiTaskConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.loss_fn = MultiTaskLoss(config)
        
    def train_epoch(
        self,
        dataloader,
        optimizer,
        scheduler=None
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = {"total": 0.0}
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            labels = {
                task: batch[f"{task}_labels"].to(self.device)
                for task in ["event", "negation", "temporal"]
                if f"{task}_labels" in batch
            }
            
            optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask, task="all")
            loss, task_losses = self.loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            total_losses["total"] += loss.item()
            for task, task_loss in task_losses.items():
                total_losses[task] = total_losses.get(task, 0.0) + task_loss.item()
        
        n_batches = len(dataloader)
        return {k: v / n_batches for k, v in total_losses.items()}
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        
        task_correct = {"event": 0, "negation": 0, "temporal": 0}
        task_total = {"event": 0, "negation": 0, "temporal": 0}
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, task="all")
                
                for task in ["event", "negation", "temporal"]:
                    if f"{task}_labels" in batch:
                        labels = batch[f"{task}_labels"].to(self.device)
                        preds = outputs[task].argmax(dim=1)
                        task_correct[task] += (preds == labels).sum().item()
                        task_total[task] += len(labels)
        
        accuracies = {}
        for task in ["event", "negation", "temporal"]:
            if task_total[task] > 0:
                accuracies[f"{task}_accuracy"] = task_correct[task] / task_total[task]
        
        return accuracies
```

### 5. Contrastive Learning for Clinical Embeddings

Learn robust sentence embeddings for clinical text.

```python
# cedars/training/contrastive.py
"""
Contrastive Learning for Clinical Embeddings

Train clinical text embeddings using contrastive learning,
useful for similarity search and few-shot learning scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning."""
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    embedding_dim: int = 256
    temperature: float = 0.07
    max_length: int = 256
    learning_rate: float = 2e-5
    batch_size: int = 32
    

class ClinicalSentenceEncoder(nn.Module):
    """Sentence encoder with projection head for contrastive learning."""
    
    def __init__(self, config: ContrastiveConfig):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, config.embedding_dim)
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_projection: bool = True
    ) -> torch.Tensor:
        """
        Encode sentences.
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            return_projection: If True, return projected embeddings
            
        Returns:
            Sentence embeddings
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        token_embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * attention_mask_expanded, 1) / torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        
        if return_projection:
            return F.normalize(self.projection(pooled), p=2, dim=1)
        return pooled


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss for clinical text."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            embeddings: Normalized embeddings (batch_size, embedding_dim)
            labels: Class labels (batch_size,)
            
        Returns:
            Scalar loss
        """
        batch_size = embeddings.shape[0]
        
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=mask.device)
        mask = mask * logits_mask
        
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)
        
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask_sum
        
        loss = -mean_log_prob.mean()
        return loss


class ClinicalContrastiveTrainer:
    """Trainer for clinical contrastive learning."""
    
    def __init__(
        self,
        config: ContrastiveConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.config = config
        self.device = device
        self.model = ClinicalSentenceEncoder(config).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.loss_fn = SupConLoss(config.temperature)
        
    def create_pairs(
        self,
        sentences: List[str],
        labels: List[int]
    ) -> List[Tuple[str, str, int]]:
        """
        Create positive and negative pairs for training.
        
        Returns:
            List of (sentence1, sentence2, is_same_class)
        """
        pairs = []
        label_to_sentences = {}
        
        for sent, label in zip(sentences, labels):
            if label not in label_to_sentences:
                label_to_sentences[label] = []
            label_to_sentences[label].append(sent)
        
        for label, sents in label_to_sentences.items():
            for i in range(len(sents)):
                for j in range(i + 1, len(sents)):
                    pairs.append((sents[i], sents[j], 1))
        
        all_labels = list(label_to_sentences.keys())
        for i, label1 in enumerate(all_labels):
            for label2 in all_labels[i + 1:]:
                for sent1 in label_to_sentences[label1][:10]:
                    for sent2 in label_to_sentences[label2][:10]:
                        pairs.append((sent1, sent2, 0))
        
        return pairs
    
    def train(
        self,
        sentences: List[str],
        labels: List[int],
        num_epochs: int = 10
    ):
        """Train the contrastive model."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        pairs = self.create_pairs(sentences, labels)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            
            np.random.shuffle(pairs)
            
            for i in range(0, len(pairs), self.config.batch_size):
                batch_pairs = pairs[i:i + self.config.batch_size]
                
                texts = [p[0] for p in batch_pairs] + [p[1] for p in batch_pairs]
                pair_labels = torch.tensor(
                    [p[2] for p in batch_pairs] * 2,
                    device=self.device
                )
                
                encoded = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                
                embeddings = self.model(input_ids, attention_mask)
                loss = self.loss_fn(embeddings, pair_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(pairs) // self.config.batch_size)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
    
    def encode(self, sentences: List[str]) -> np.ndarray:
        """Encode sentences to embeddings."""
        self.model.eval()
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sentences), self.config.batch_size):
                batch = sentences[i:i + self.config.batch_size]
                
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                
                embeddings = self.model(input_ids, attention_mask)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def find_similar(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find most similar sentences to query."""
        query_emb = self.encode([query])
        corpus_emb = self.encode(corpus)
        
        similarities = np.dot(corpus_emb, query_emb.T).squeeze()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(corpus[i], similarities[i]) for i in top_indices]
```

## Integration with CEDARS

### Adding Training Module to Project Structure

```
cedars/
├── app/
│   ├── __init__.py
│   ├── nlpprocessor.py
│   └── ...
├── training/
│   ├── __init__.py
│   ├── event_classifier.py
│   ├── medical_ner.py
│   ├── active_learning.py
│   ├── multitask.py
│   ├── contrastive.py
│   └── utils/
│       ├── __init__.py
│       ├── data_loading.py
│       └── evaluation.py
└── models/
    └── .gitkeep
```

### Dependencies to Add

```toml
# Add to pyproject.toml [tool.poetry.dependencies]
torch = ">=2.0.0"
transformers = ">=4.30.0"
scikit-learn = ">=1.3.0"
accelerate = ">=0.20.0"
datasets = ">=2.12.0"
evaluate = ">=0.4.0"
```

## Recommended Training Workflow

1. **Data Collection Phase**
   - Use CEDARS annotation interface to collect labeled examples
   - Export annotations periodically for model training

2. **Initial Model Training**
   - Start with event classifier using PubMedBERT
   - Evaluate on held-out test set

3. **Active Learning Loop**
   - Deploy model for uncertainty sampling
   - Prioritize annotation of uncertain samples
   - Retrain periodically with new annotations

4. **Model Deployment**
   - Integrate trained model with PINES API
   - Use predictions to pre-filter annotations
   - Monitor model performance over time

## Evaluation Metrics

Key metrics to track:

| Metric | Target | Description |
|--------|--------|-------------|
| F1 Score | >0.85 | Balance of precision/recall |
| Recall | >0.90 | Minimize missed events |
| AUC-ROC | >0.90 | Overall discrimination |
| Annotation Time Savings | >50% | Efficiency gain |

## Next Steps

1. Create training data export utility in CEDARS
2. Set up training infrastructure (GPU server)
3. Implement baseline event classifier
4. Add active learning to annotation workflow
5. Evaluate and iterate on model performance
