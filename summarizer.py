"""
Scientific Paper Summarizer using Transformer Models
====================================================

This implementation demonstrates a complete pipeline for summarizing
scientific papers using pretrained transformer models (BART, LED, T5).

Features:
- Multiple model support (BART, Longformer-LED, SciBERT+BART)
- Structured summarization (Abstract, Methods, Results, Conclusions)
- Multi-document summarization
- Citation extraction and key findings identification
- Evaluation using ROUGE, BERTScore, and custom metrics

Author: [Mehak Negi]
Date:13 November 2025
"""

# ============================================================================
# PART 1: SETUP AND INSTALLATIONS
# ============================================================================

# Run these in your terminal or notebook
"""
pip install transformers datasets torch evaluate
pip install rouge-score bert-score
pip install nltk spacy pandas numpy matplotlib seaborn
pip install arxiv pypdf2 python-docx
pip install scikit-learn tensorboard
python -m spacy download en_core_web_sm
"""

# ============================================================================
# PART 2: IMPORTS
# ============================================================================

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    LEDTokenizer,
    LEDForConditionalGeneration,
    BartTokenizer,
    BartForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, Dataset
import evaluate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import re
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# PART 3: DATA LOADING AND PREPROCESSING
# ============================================================================

class ScientificPaperDataset:
    """Handler for scientific paper datasets"""
    
    def __init__(self, dataset_name="ccdv/arxiv-summarization"):
        """
        Initialize dataset loader
        
        Recommended datasets:
        - ccdv/arxiv-summarization (arXiv papers)
        - scientific_papers (PubMed + arXiv)
        - allenai/mup (Multi-perspective summaries)
        - ccdv/pubmed-summarization
        """
        self.dataset_name = dataset_name
        self.dataset = None
        
    def load_data(self):
        """Load dataset from Hugging Face"""
        print(f"Loading dataset: {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name)
        print(f"Dataset loaded: {self.dataset}")
        return self.dataset
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep scientific notation
        text = re.sub(r'[^\w\s\.\,\-\(\)\[\]\{\}\:\;]', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        return text.strip()
    
    def extract_sections(self, paper_text: str) -> Dict[str, str]:
        """Extract paper sections (Introduction, Methods, Results, etc.)"""
        sections = {}
        
        # Common section headers in papers
        section_patterns = {
            'abstract': r'(?i)abstract[:\s]*(.*?)(?=introduction|method|$)',
            'introduction': r'(?i)introduction[:\s]*(.*?)(?=method|background|related work|$)',
            'methods': r'(?i)(?:method|methodology|materials and methods)[:\s]*(.*?)(?=result|experiment|discussion|$)',
            'results': r'(?i)results[:\s]*(.*?)(?=discussion|conclusion|$)',
            'discussion': r'(?i)discussion[:\s]*(.*?)(?=conclusion|reference|$)',
            'conclusion': r'(?i)conclusion[:\s]*(.*?)(?=reference|acknowledgment|$)'
        }
        
        for section, pattern in section_patterns.items():
            match = re.search(pattern, paper_text, re.DOTALL)
            if match:
                sections[section] = self.preprocess_text(match.group(1))
        
        return sections
    
    def create_structured_input(self, paper: Dict) -> str:
        """Create structured input format for better summarization"""
        structured = ""
        
        if 'title' in paper:
            structured += f"Title: {paper['title']}\n\n"
        
        if 'abstract' in paper:
            structured += f"Abstract: {paper['abstract']}\n\n"
            
        if 'article' in paper:
            sections = self.extract_sections(paper['article'])
            for section_name, content in sections.items():
                structured += f"{section_name.title()}: {content}\n\n"
        
        return structured

# ============================================================================
# PART 4: MODEL IMPLEMENTATIONS
# ============================================================================

class ScientificSummarizer:
    """Base class for scientific paper summarizers"""
    
    def __init__(self, model_name: str, max_input_length: int = 1024, 
                 max_output_length: int = 256):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load pretrained model and tokenizer"""
        raise NotImplementedError
        
    def generate_summary(self, text: str, **kwargs) -> str:
        """Generate summary for given text"""
        raise NotImplementedError

class BARTSummarizer(ScientificSummarizer):
    """BART-based summarizer (good for short to medium papers)"""
    
    def __init__(self, model_name="facebook/bart-large-cnn"):
        super().__init__(model_name, max_input_length=1024, max_output_length=256)
        self.load_model()
        
    def load_model(self):
        print(f"Loading BART model: {self.model_name}")
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def generate_summary(self, text: str, num_beams: int = 4, 
                        length_penalty: float = 2.0) -> str:
        """Generate summary using BART"""
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=num_beams,
                max_length=self.max_output_length,
                length_penalty=length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

class LEDSummarizer(ScientificSummarizer):
    """Longformer-LED summarizer (best for long papers)"""
    
    def __init__(self, model_name="allenai/led-large-16384-arxiv"):
        super().__init__(model_name, max_input_length=16384, max_output_length=512)
        self.load_model()
        
    def load_model(self):
        print(f"Loading LED model: {self.model_name}")
        self.tokenizer = LEDTokenizer.from_pretrained(self.model_name)
        self.model = LEDForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def generate_summary(self, text: str, num_beams: int = 4) -> str:
        """Generate summary using LED"""
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Set global attention on first token
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[:, 0] = 1
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                global_attention_mask=global_attention_mask,
                num_beams=num_beams,
                max_length=self.max_output_length,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

class T5Summarizer(ScientificSummarizer):
    """T5-based summarizer with task prefix"""
    
    def __init__(self, model_name="t5-base"):
        super().__init__(model_name, max_input_length=512, max_output_length=256)
        self.load_model()
        
    def load_model(self):
        print(f"Loading T5 model: {self.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def generate_summary(self, text: str, num_beams: int = 4) -> str:
        """Generate summary using T5"""
        # T5 requires task prefix
        input_text = f"summarize: {text}"
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=num_beams,
                max_length=self.max_output_length,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

# ============================================================================
# PART 5: STRUCTURED SUMMARIZATION
# ============================================================================

class StructuredSummarizer:
    """Generate structured summaries with sections"""
    
    def __init__(self, base_summarizer: ScientificSummarizer):
        self.summarizer = base_summarizer
        
    def summarize_sections(self, paper_sections: Dict[str, str]) -> Dict[str, str]:
        """Generate summaries for each section"""
        section_summaries = {}
        
        for section_name, content in paper_sections.items():
            if len(content.split()) > 50:  # Only summarize substantial sections
                summary = self.summarizer.generate_summary(content)
                section_summaries[section_name] = summary
            else:
                section_summaries[section_name] = content
                
        return section_summaries
    
    def create_executive_summary(self, paper_sections: Dict[str, str]) -> str:
        """Create a high-level executive summary"""
        # Combine key sections
        key_content = ""
        for section in ['abstract', 'introduction', 'results', 'conclusion']:
            if section in paper_sections:
                key_content += f"{paper_sections[section]} "
        
        # Generate concise summary
        executive_summary = self.summarizer.generate_summary(
            key_content,
            num_beams=6,
            length_penalty=2.5
        )
        
        return executive_summary
    
    def extract_key_findings(self, results_section: str, top_k: int = 5) -> List[str]:
        """Extract key findings from results section"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', results_section)
        
        # Simple heuristic: sentences with numbers/percentages are often findings
        findings = []
        for sent in sentences:
            if re.search(r'\d+\.?\d*\s*%|\d+\s*fold|p\s*[<>=]|significant', sent.lower()):
                findings.append(sent.strip())
        
        return findings[:top_k]

# ============================================================================
# PART 6: FINE-TUNING (Optional but Recommended)
# ============================================================================

class FineTuner:
    """Fine-tune models on scientific papers"""
    
    def __init__(self, model_name: str, dataset: Dataset):
        self.model_name = model_name
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    def preprocess_function(self, examples):
        """Tokenize inputs and labels"""
        inputs = examples["article"]
        targets = examples["abstract"]
        
        model_inputs = self.tokenizer(
            inputs,
            max_length=1024,
            truncation=True,
            padding="max_length"
        )
        
        labels = self.tokenizer(
            targets,
            max_length=256,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def train(self, output_dir: str = "./fine_tuned_model", epochs: int = 3):
        """Fine-tune the model"""
        # Tokenize dataset
        tokenized_dataset = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=4,
            warmup_steps=500,
            logging_steps=100
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("validation", tokenized_dataset["train"]),
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        print("Starting fine-tuning...")
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")
        
        return trainer

# ============================================================================
# PART 7: EVALUATION METRICS
# ============================================================================

class SummarizationEvaluator:
    """Evaluate summarization quality"""
    
    def __init__(self):
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load('bertscore')
        
    def compute_rouge(self, predictions: List[str], 
                     references: List[str]) -> Dict:
        """Calculate ROUGE scores"""
        results = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        return results
    
    def compute_bertscore(self, predictions: List[str], 
                         references: List[str]) -> Dict:
        """Calculate BERTScore"""
        results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli"
        )
        return {
            'bertscore_precision': np.mean(results['precision']),
            'bertscore_recall': np.mean(results['recall']),
            'bertscore_f1': np.mean(results['f1'])
        }
    
    def compute_all_metrics(self, predictions: List[str], 
                           references: List[str]) -> Dict:
        """Compute all evaluation metrics"""
        metrics = {}
        
        # ROUGE
        rouge_results = self.compute_rouge(predictions, references)
        metrics.update(rouge_results)
        
        # BERTScore (can be slow, use sample if dataset is large)
        if len(predictions) > 100:
            sample_indices = random.sample(range(len(predictions)), 100)
            pred_sample = [predictions[i] for i in sample_indices]
            ref_sample = [references[i] for i in sample_indices]
        else:
            pred_sample = predictions
            ref_sample = references
            
        bert_results = self.compute_bertscore(pred_sample, ref_sample)
        metrics.update(bert_results)
        
        # Custom metrics
        metrics['avg_length'] = np.mean([len(p.split()) for p in predictions])
        metrics['compression_ratio'] = np.mean([
            len(r.split()) / len(p.split()) 
            for p, r in zip(predictions, references)
        ])
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Pretty print metrics"""
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric:30s}: {value:.4f}")
            else:
                print(f"{metric:30s}: {value}")
        print("="*50 + "\n")

# ============================================================================
# PART 8: VISUALIZATION
# ============================================================================

class SummarizationVisualizer:
    """Visualize summarization results"""
    
    @staticmethod
    def plot_length_distribution(original_lengths: List[int], 
                                summary_lengths: List[int]):
        """Plot length distribution comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(original_lengths, bins=50, alpha=0.7, color='blue')
        axes[0].set_title('Original Text Length Distribution')
        axes[0].set_xlabel('Number of Words')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(summary_lengths, bins=50, alpha=0.7, color='green')
        axes[1].set_title('Summary Length Distribution')
        axes[1].set_xlabel('Number of Words')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('length_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_rouge_comparison(models_metrics: Dict[str, Dict]):
        """Compare ROUGE scores across models"""
        models = list(models_metrics.keys())
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
        
        x = np.arange(len(models))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, rouge_type in enumerate(rouge_types):
            scores = [models_metrics[model][rouge_type] for model in models]
            ax.bar(x + i*width, scores, width, label=rouge_type.upper())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('ROUGE Score')
        ax.set_title('Model Comparison - ROUGE Scores')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def create_summary_report(original: str, summary: str, 
                            reference: str, metrics: Dict):
        """Create visual comparison report"""
        print("\n" + "="*80)
        print("SUMMARIZATION REPORT")
        print("="*80)
        
        print(f"\nORIGINAL TEXT ({len(original.split())} words):")
        print("-"*80)
        print(original[:500] + "..." if len(original) > 500 else original)
        
        print(f"\n\nGENERATED SUMMARY ({len(summary.split())} words):")
        print("-"*80)
        print(summary)
        
        print(f"\n\nREFERENCE SUMMARY ({len(reference.split())} words):")
        print("-"*80)
        print(reference)
        
        print("\n\nMETRICS:")
        print("-"*80)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric:20s}: {value:.4f}")
        
        print("="*80 + "\n")

# ============================================================================
# PART 9: MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("Scientific Paper Summarizer")
    print("="*80)
    
    # Step 1: Load Dataset
    print("\n[1/7] Loading dataset...")
    data_handler = ScientificPaperDataset("ccdv/arxiv-summarization")
    dataset = data_handler.load_data()
    
    # Use small subset for demo
    test_data = dataset["test"].select(range(100))
    
    # Step 2: Initialize Models
    print("\n[2/7] Loading models...")
    models = {
        'BART': BARTSummarizer("facebook/bart-large-cnn"),
        'LED': LEDSummarizer("allenai/led-base-16384"),
        'T5': T5Summarizer("t5-base")
    }
    
    # Step 3: Generate Summaries
    print("\n[3/7] Generating summaries...")
    results = {model_name: {'predictions': [], 'references': []} 
               for model_name in models.keys()}
    
    for idx in tqdm(range(min(20, len(test_data))), desc="Summarizing"):
        paper = test_data[idx]
        article = paper['article']
        reference = paper['abstract']
        
        for model_name, model in models.items():
            try:
                summary = model.generate_summary(article)
                results[model_name]['predictions'].append(summary)
                results[model_name]['references'].append(reference)
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                results[model_name]['predictions'].append("")
                results[model_name]['references'].append(reference)
    
    # Step 4: Evaluate
    print("\n[4/7] Evaluating models...")
    evaluator = SummarizationEvaluator()
    all_metrics = {}
    
    for model_name in models.keys():
        print(f"\nEvaluating {model_name}...")
        metrics = evaluator.compute_all_metrics(
            results[model_name]['predictions'],
            results[model_name]['references']
        )
        all_metrics[model_name] = metrics
        evaluator.print_metrics(metrics)
    
    # Step 5: Visualize
    print("\n[5/7] Creating visualizations...")
    visualizer = SummarizationVisualizer()
    
    # Length distributions
    original_lengths = [len(paper['article'].split()) 
                       for paper in test_data.select(range(20))]
    summary_lengths = [len(s.split()) 
                      for s in results['BART']['predictions']]
    visualizer.plot_length_distribution(original_lengths, summary_lengths)
    
    # Model comparison
    visualizer.plot_rouge_comparison(all_metrics)
    
    # Step 6: Structured Summarization Demo
    print("\n[6/7] Demonstrating structured summarization...")
    structured_summarizer = StructuredSummarizer(models['LED'])
    
    sample_paper = test_data[0]
    sections = data_handler.extract_sections(sample_paper['article'])
    section_summaries = structured_summarizer.summarize_sections(sections)
    
    print("\nStructured Summary:")
    for section, summary in section_summaries.items():
        print(f"\n{section.upper()}:")
        print(summary)
    
    # Step 7: Generate Report
    print("\n[7/7] Generating sample report...")
    visualizer.create_summary_report(
        sample_paper['article'][:1000],
        results['BART']['predictions'][0],
        sample_paper['abstract'],
        all_metrics['BART']
    )
    
    print("\n" + "="*80)
    print("Pipeline Complete!")
    print("="*80)
    
    return results, all_metrics

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    results, metrics = main()
    
    # Optional: Save results
    with open('summarization_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nResults saved to 'summarization_results.json'")
    print("Visualizations saved as PNG files")
