# Scientific Paper Summarizer - Quick Start Notebook
# This notebook provides a streamlined workflow for immediate experimentation

# %% [markdown]
# # Scientific Paper Summarizer - Quick Start
# 
# This notebook demonstrates:
# 1. Loading the dataset
# 2. Running inference with pretrained models
# 3. Evaluating results
# 4. Visualizing outputs
# 
# **Estimated runtime: 15-20 minutes on GPU**

# %% Setup
# Install required packages (uncomment if needed)
# !pip install transformers datasets evaluate rouge-score bert-score
# !pip install torch pandas matplotlib seaborn

import warnings
warnings.filterwarnings('ignore')

# %% Imports
from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
import numpy as np
from tqdm.auto import tqdm

print("✓ Imports successful")

# %% Configuration
CONFIG = {
    'dataset': 'ccdv/arxiv-summarization',
    'num_samples': 50,  # Start small for quick testing
    'models': {
        'bart': 'facebook/bart-large-cnn',
        'led': 'allenai/led-base-16384',
        't5': 't5-base'
    },
    'max_input_length': 1024,
    'max_output_length': 256,
    'num_beams': 4
}

print("✓ Configuration set")

# %% Step 1: Load Dataset
print("\n📚 Loading dataset...")

dataset = load_dataset(CONFIG['dataset'])
test_data = dataset['test'].select(range(CONFIG['num_samples']))

print(f"✓ Loaded {len(test_data)} test samples")
print(f"Sample paper length: {len(test_data[0]['article'].split())} words")
print(f"Sample abstract length: {len(test_data[0]['abstract'].split())} words")

# %% Step 2: Explore Data
print("\n🔍 Data Exploration...")

# Calculate statistics
stats = pd.DataFrame({
    'article_length': [len(p['article'].split()) for p in test_data],
    'abstract_length': [len(p['abstract'].split()) for p in test_data]
})

print(stats.describe())

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(stats['article_length'], bins=20, color='skyblue', edgecolor='black')
axes[0].set_title('Article Length Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Words')
axes[0].set_ylabel('Frequency')

axes[1].hist(stats['abstract_length'], bins=20, color='lightcoral', edgecolor='black')
axes[1].set_title('Abstract Length Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Words')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualization saved as 'data_exploration.png'")

# %% Step 3: Initialize Models
print("\n🤖 Loading models...")

# BART - Good for general use
print("Loading BART...")
bart_summarizer = pipeline(
    "summarization",
    model=CONFIG['models']['bart'],
    device=0 if __import__('torch').cuda.is_available() else -1
)
print("✓ BART loaded")

# T5 - Fast and efficient
print("Loading T5...")
t5_summarizer = pipeline(
    "summarization",
    model=CONFIG['models']['t5'],
    device=0 if __import__('torch').cuda.is_available() else -1
)
print("✓ T5 loaded")

print("\n✓ All models ready!")

# %% Step 4: Generate Summaries
print("\n📝 Generating summaries...")

results = {
    'references': [],
    'bart_summaries': [],
    't5_summaries': [],
    'articles': []
}

for paper in tqdm(test_data, desc="Summarizing papers"):
    article = paper['article'][:CONFIG['max_input_length'] * 6]  # Approximate token limit
    reference = paper['abstract']
    
    # BART summary
    try:
        bart_out = bart_summarizer(
            article,
            max_length=CONFIG['max_output_length'],
            min_length=50,
            num_beams=CONFIG['num_beams'],
            early_stopping=True
        )
        bart_summary = bart_out[0]['summary_text']
    except Exception as e:
        bart_summary = "Error generating summary"
        print(f"BART error: {e}")
    
    # T5 summary
    try:
        t5_out = t5_summarizer(
            article,
            max_length=CONFIG['max_output_length'],
            min_length=50,
            num_beams=CONFIG['num_beams']
        )
        t5_summary = t5_out[0]['summary_text']
    except Exception as e:
        t5_summary = "Error generating summary"
        print(f"T5 error: {e}")
    
    results['references'].append(reference)
    results['bart_summaries'].append(bart_summary)
    results['t5_summaries'].append(t5_summary)
    results['articles'].append(article[:500])  # Store snippet

print("✓ Summaries generated")

# %% Step 5: Evaluate with ROUGE
print("\n📊 Evaluating with ROUGE scores...")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge_scores(predictions, references):
    """Calculate average ROUGE scores"""
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)
    
    return {k: np.mean(v) for k, v in scores.items()}

# Calculate scores for each model
bart_scores = calculate_rouge_scores(results['bart_summaries'], results['references'])
t5_scores = calculate_rouge_scores(results['t5_summaries'], results['references'])

# Create results dataframe
eval_results = pd.DataFrame({
    'Model': ['BART', 'T5'],
    'ROUGE-1': [bart_scores['rouge1'], t5_scores['rouge1']],
    'ROUGE-2': [bart_scores['rouge2'], t5_scores['rouge2']],
    'ROUGE-L': [bart_scores['rougeL'], t5_scores['rougeL']]
})

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(eval_results.to_string(index=False))
print("="*60)

# %% Step 6: Visualize Results
print("\n📈 Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Bar plot
x = np.arange(len(eval_results))
width = 0.25

axes[0].bar(x - width, eval_results['ROUGE-1'], width, label='ROUGE-1', color='#3498db')
axes[0].bar(x, eval_results['ROUGE-2'], width, label='ROUGE-2', color='#e74c3c')
axes[0].bar(x + width, eval_results['ROUGE-L'], width, label='ROUGE-L', color='#2ecc71')

axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
axes[0].set_title('Model Comparison - ROUGE Scores', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(eval_results['Model'])
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Summary length comparison
summary_lengths = {
    'BART': [len(s.split()) for s in results['bart_summaries']],
    'T5': [len(s.split()) for s in results['t5_summaries']],
    'Reference': [len(s.split()) for s in results['references']]
}

axes[1].boxplot(summary_lengths.values(), labels=summary_lengths.keys())
axes[1].set_ylabel('Number of Words', fontsize=12, fontweight='bold')
axes[1].set_title('Summary Length Distribution', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualization saved as 'evaluation_results.png'")

# %% Step 7: Sample Outputs
print("\n📄 Sample Outputs:")
print("="*80)

sample_idx = 0
print(f"\n[Paper {sample_idx + 1}]")
print("\nORIGINAL ARTICLE (first 300 chars):")
print("-"*80)
print(results['articles'][sample_idx][:300] + "...")

print("\n\nREFERENCE ABSTRACT:")
print("-"*80)
print(results['references'][sample_idx])

print("\n\nBART SUMMARY:")
print("-"*80)
print(results['bart_summaries'][sample_idx])

print("\n\nT5 SUMMARY:")
print("-"*80)
print(results['t5_summaries'][sample_idx])

print("\n" + "="*80)

# %% Step 8: Detailed Error Analysis
print("\n🔬 Error Analysis...")

def analyze_summaries(summaries, references, model_name):
    """Analyze summary quality"""
    analysis = {
        'avg_length': np.mean([len(s.split()) for s in summaries]),
        'length_std': np.std([len(s.split()) for s in summaries]),
        'compression_ratio': np.mean([
            len(r.split()) / max(len(s.split()), 1) 
            for s, r in zip(summaries, references)
        ])
    }
    
    print(f"\n{model_name} Analysis:")
    print(f"  Average length: {analysis['avg_length']:.1f} words")
    print(f"  Length std dev: {analysis['length_std']:.1f}")
    print(f"  Compression ratio: {analysis['compression_ratio']:.2f}x")
    
    return analysis

bart_analysis = analyze_summaries(
    results['bart_summaries'], 
    results['references'], 
    'BART'
)

t5_analysis = analyze_summaries(
    results['t5_summaries'], 
    results['references'], 
    'T5'
)

# %% Step 9: Save Results
print("\n💾 Saving results...")

# Save to CSV
results_df = pd.DataFrame({
    'article_snippet': results['articles'],
    'reference': results['references'],
    'bart_summary': results['bart_summaries'],
    't5_summary': results['t5_summaries']
})

results_df.to_csv('summarization_results.csv', index=False)
print("✓ Results saved to 'summarization_results.csv'")

# Save metrics
eval_results.to_csv('evaluation_metrics.csv', index=False)
print("✓ Metrics saved to 'evaluation_metrics.csv'")

# %% Step 10: Next Steps
print("\n🚀 Next Steps:")
print("-"*80)
print("1. Fine-tune models on the full dataset")
print("2. Implement LED for longer papers")
print("3. Add structured summarization (section-wise)")
print("4. Experiment with different hyperparameters")
print("5. Conduct human evaluation")
print("6. Deploy as API or web application")
print("-"*80)

# %% Bonus: Interactive Summary Comparison
def compare_summaries(index):
    """Interactive comparison function"""
    print("\n" + "="*80)
    print(f"COMPARISON FOR PAPER {index + 1}")
    print("="*80)
    
    print("\n📰 REFERENCE ABSTRACT:")
    print(results['references'][index])
    
    print("\n🤖 BART SUMMARY:")
    print(results['bart_summaries'][index])
    
    print("\n🤖 T5 SUMMARY:")
    print(results['t5_summaries'][index])
    
    # Calculate individual ROUGE scores
    bart_score = scorer.score(
        results['references'][index], 
        results['bart_summaries'][index]
    )
    t5_score = scorer.score(
        results['references'][index], 
        results['t5_summaries'][index]
    )
    
    print("\n📊 SCORES:")
    print(f"BART - ROUGE-L: {bart_score['rougeL'].fmeasure:.4f}")
    print(f"T5   - ROUGE-L: {t5_score['rougeL'].fmeasure:.4f}")
    print("="*80)

# Example usage
compare_summaries(0)

print("\n✅ Notebook execution complete!")
print("All results saved. Review the generated CSV files and PNG visualizations.")
