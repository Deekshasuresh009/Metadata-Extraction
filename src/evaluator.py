"""
Evaluator Module
Calculates Per-Field Recall according to the official assignment specification:
True = Number of exact matches
False = Number of mismatches or unextracted fields
Recall = True / (True + False)
"""

import pandas as pd
from typing import Dict, List, Any

TARGET_FIELDS = [
    "Aggrement Value",
    "Aggrement Start Date",
    "Aggrement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two"
]

class MetadataEvaluator:
    """Evaluates extracted metadata against ground truth datasets."""

    @staticmethod
    def calculate_recall(ground_truth_df: pd.DataFrame, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes per-field and overall recall metrics.
        
        Args:
            ground_truth_df (pd.DataFrame): Expected ground truth table.
            predictions_df (pd.DataFrame): System predicted metadata table.
            
        Returns:
            dict: Per-field True/False counts, Recall scores, and Overall Mean Recall.
        """
        # Merge on File Name
        merged = pd.merge(ground_truth_df, predictions_df, on="File Name", suffixes=("_gt", "_pred"))
        
        results = {}
        total_true = 0
        total_false = 0
        
        for field in TARGET_FIELDS:
            gt_col = f"{field}_gt"
            pred_col = f"{field}_pred"
            
            if gt_col not in merged.columns or pred_col not in merged.columns:
                continue
                
            true_count = 0
            false_count = 0
            
            for _, row in merged.iterrows():
                gt_val = str(row[gt_col]).strip() if pd.notna(row[gt_col]) else ""
                pred_val = str(row[pred_col]).strip() if pd.notna(row[pred_col]) else ""
                
                # Check exact match
                if gt_val == pred_val and gt_val != "":
                    true_count += 1
                elif gt_val == "" and pred_val == "":
                    true_count += 1
                else:
                    false_count += 1
                    
            recall = true_count / (true_count + false_count) if (true_count + false_count) > 0 else 0.0
            results[field] = {
                "True": true_count,
                "False": false_count,
                "Recall": round(recall, 4)
            }
            total_true += true_count
            total_false += false_count
            
        overall_recall = total_true / (total_true + total_false) if (total_true + total_false) > 0 else 0.0
        results["Overall"] = {
            "Total_True": total_true,
            "Total_False": total_false,
            "Recall": round(overall_recall, 4)
        }
        return results
