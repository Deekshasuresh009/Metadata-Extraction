"""
USEReady AI/ML Internship Assignment: Metadata Extraction from Documents
Main CLI Runner & Evaluation Entrypoint

Usage:
  python main.py                           # Run extraction on test dataset (default)
  python main.py --input data/train        # Run extraction on training dataset
  python main.py --file path/to/doc.docx   # Run extraction on a single document
  python main.py --evaluate data/test.csv  # Run extraction on test set and evaluate recall
"""

import os
import sys
import glob
import argparse
import json
import pandas as pd

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from src.predictor import MetadataPredictor
from src.evaluator import MetadataEvaluator

def run_pipeline(input_path: str, evaluate_csv: str = None, output_csv: str = None):
    print("=" * 80)
    print(" USEReady Contract Metadata Extraction Pipeline")
    print(" Model: deepset/minilm-uncased-squad2 (Extractive QA Transformer)")
    print("=" * 80)

    predictor = MetadataPredictor()
    files_to_process = []

    if os.path.isfile(input_path):
        files_to_process = [input_path]
    elif os.path.isdir(input_path):
        for ext in ("*.docx", "*.png", "*.jpg", "*.jpeg"):
            files_to_process.extend(glob.glob(os.path.join(input_path, ext)))
        files_to_process = sorted(files_to_process)
    else:
        print(f"[ERROR] Input path not found: {input_path}")
        sys.exit(1)

    print(f"\n[INFO] Found {len(files_to_process)} document(s) to process from: {input_path}\n")

    results = []
    for idx, f_path in enumerate(files_to_process, 1):
        file_name = os.path.basename(f_path)
        print(f"[{idx}/{len(files_to_process)}] Processing: {file_name} ...")
        
        try:
            pred = predictor.process_document(f_path)
            row = {
                "File Name": file_name,
                "Aggrement Value": pred.get("Aggrement Value", ""),
                "Aggrement Start Date": pred.get("Aggrement Start Date", ""),
                "Aggrement End Date": pred.get("Aggrement End Date", ""),
                "Renewal Notice (Days)": pred.get("Renewal Notice (Days)", ""),
                "Party One": pred.get("Party One", ""),
                "Party Two": pred.get("Party Two", "")
            }
            results.append(row)
            print(f"    -> Value: {row['Aggrement Value']} | Start: {row['Aggrement Start Date']} | End: {row['Aggrement End Date']} | Notice: {row['Renewal Notice (Days)']} days")
            print(f"    -> Party One: {row['Party One']}")
            print(f"    -> Party Two: {row['Party Two']}\n")
        except Exception as e:
            print(f"    -> [FAILED]: {e}\n")

    df_preds = pd.DataFrame(results)

    # Save output if specified
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df_preds.to_csv(output_csv, index=False)
        print(f"[INFO] Saved predictions to: {output_csv}\n")

    # Display summary table
    print("=" * 80)
    print(" PREDICTION RESULTS SUMMARY")
    print("=" * 80)
    if not df_preds.empty:
        print(df_preds.to_string(index=False))
    print("=" * 80)

    # Evaluate against ground truth if requested
    if evaluate_csv and os.path.exists(evaluate_csv):
        print("\n" + "=" * 80)
        print(f" OFFICIAL EXACT-MATCH EVALUATION (Ground Truth: {evaluate_csv})")
        print("=" * 80)
        
        gt_df = pd.read_csv(evaluate_csv)
        
        # Normalize file names for comparison
        def norm_id(name):
            s = os.path.basename(str(name))
            return s.replace('.pdf.docx', '').replace('.docx', '').replace('.png', '').replace('.pdf', '').strip()

        preds_clean = df_preds.copy()
        preds_clean['File Name'] = preds_clean['File Name'].apply(norm_id)
        gt_clean = gt_df.copy()
        gt_clean['File Name'] = gt_clean['File Name'].apply(norm_id)

        eval_metrics = MetadataEvaluator.calculate_recall(gt_clean, preds_clean)
        
        print(f"{'Target Field':25} | {'True':6} | {'False':6} | {'Recall':8}")
        print("-" * 55)
        for field in ["Aggrement Value", "Aggrement Start Date", "Aggrement End Date", "Renewal Notice (Days)", "Party One", "Party Two"]:
            m = eval_metrics.get(field, {"True": 0, "False": 0, "Recall": 0.0})
            print(f"{field:25} | {m['True']:<6} | {m['False']:<6} | {m['Recall']*100:5.1f}%")
        print("-" * 55)
        overall = eval_metrics.get("Overall", {"Total_True": 0, "Total_False": 0, "Recall": 0.0})
        print(f"{'OVERALL RECALL':25} | {overall['Total_True']:<6} | {overall['Total_False']:<6} | {overall['Recall']*100:5.1f}%")
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="USEReady Contract Metadata Extraction CLI")
    parser.add_argument("--input", "-i", default=os.path.join(BASE_DIR, "data", "test"), help="Path to input directory of documents (default: data/test)")
    parser.add_argument("--file", "-f", default=None, help="Path to a single document (.docx or .png)")
    parser.add_argument("--evaluate", "-e", default=None, help="Path to ground truth CSV for exact-match recall evaluation")
    parser.add_argument("--output", "-o", default=None, help="Path to output CSV to save predictions")

    args = parser.parse_args()
    target_input = args.file if args.file else args.input
    run_pipeline(input_path=target_input, evaluate_csv=args.evaluate, output_csv=args.output)

if __name__ == "__main__":
    main()
