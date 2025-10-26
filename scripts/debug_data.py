"""
Debug Script - Understanding the Brain Connectivity Data
========================================================
This script helps you understand what's happening with your data
and why you got that error.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from brain_pipeline import Config, DataLoader, ConnectivityProcessor

def main():
    """
    Let's debug and understand your data step by step!
    """
    
    print("\n" + "="*70)
    print("BRAIN CONNECTIVITY DATA DEBUGGER")
    print("="*70)
    print("\nLet's understand what's happening with your data...\n")
    
    # Load configuration
    config = Config()
    data_loader = DataLoader(config)
    processor = ConnectivityProcessor()
    
    # ============================================
    # STEP 1: Load and inspect the data
    # ============================================
    print("📊 STEP 1: Loading PIOP-2 data")
    print("-" * 40)
    
    try:
        df_piop2 = data_loader.load_piop2()
        print(f"✓ Data shape: {df_piop2.shape}")
        print(f"  → {df_piop2.shape[0]} subjects (rows)")
        print(f"  → {df_piop2.shape[1]} columns total")
        print(f"  → 1 subject ID column + {df_piop2.shape[1]-1} connection columns")
        
        # Show first few column names
        print(f"\n  First column (subject ID): {df_piop2.columns[0]}")
        print(f"  Connection columns (first 3): {list(df_piop2.columns[1:4])}")
        print(f"  Connection columns (last): {df_piop2.columns[-1]}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1
    
    # ============================================
    # STEP 2: Extract regions
    # ============================================
    print("\n📍 STEP 2: Extracting brain regions")
    print("-" * 40)
    
    connection_columns = data_loader.get_connection_columns(df_piop2)
    print(f"✓ Found {len(connection_columns)} connection columns")
    
    # Show example of how connections are named
    print(f"\n  Example connection names:")
    for i in range(min(3, len(connection_columns))):
        col = connection_columns[i]
        regions = col.split('~')
        print(f"    {i+1}. '{col}'")
        print(f"       → Connects: {regions[0]} ←→ {regions[1]}")
    
    # Extract unique regions
    region_list, region_to_idx, n_regions = processor.extract_regions(connection_columns)
    print(f"\n✓ Extracted {n_regions} unique brain regions")
    
    # Show first few regions
    print(f"\n  First 5 regions:")
    for i in range(min(5, len(region_list))):
        print(f"    {i}: {region_list[i]}")
    
    print(f"\n  Last region:")
    print(f"    {n_regions-1}: {region_list[-1]}")
    
    # ============================================
    # STEP 3: Understand the dataset creation
    # ============================================
    print("\n🔧 STEP 3: Understanding dataset creation")
    print("-" * 40)
    
    print("\n💡 HERE'S WHAT HAPPENS:")
    print("  1. We have {} brain regions total".format(n_regions))
    print("  2. Each region connects to all {} other regions".format(n_regions))
    print("  3. The diagonal (self-connections) = 1.0 always")
    print("  4. When creating features, we REMOVE the diagonal")
    print("  5. So each region has {} features (connections to OTHER regions)".format(n_regions-1))
    
    # Create a small sample to demonstrate
    print("\n📝 Creating a small sample dataset (first 2 subjects)...")
    df_sample = df_piop2.head(2)
    X_sample, y_sample, subjects_sample = processor.create_dataset(
        df_sample, connection_columns
    )
    
    print(f"\n✓ Sample dataset created:")
    print(f"  → X shape: {X_sample.shape}")
    print(f"    • {X_sample.shape[0]} samples = {len(df_sample)} subjects × {n_regions} regions")
    print(f"    • {X_sample.shape[1]} features = connections to {n_regions-1} other regions")
    print(f"  → y shape: {y_sample.shape}")
    print(f"    • Labels range from 0 to {n_regions-1}")
    print(f"  → Subjects: {len(subjects_sample)} samples, {len(np.unique(subjects_sample))} unique subjects")
    
    # ============================================
    # STEP 4: Explain the original error
    # ============================================
    print("\n❌ STEP 4: Understanding your original error")
    print("-" * 40)
    
    print("\nYOUR ERROR WAS:")
    print("  ValueError: Shape of passed values is (51968, 231), indices imply (51968, 232)")
    
    print("\nWHY IT HAPPENED:")
    print(f"  • You have {n_regions} brain regions")
    print(f"  • Each region's feature vector has {n_regions-1} features (diagonal removed)")
    print(f"  • You tried to use {n_regions} region names as column headers")
    print(f"  • But you only have {n_regions-1} columns of data!")
    
    print("\nTHE FIX:")
    print("  • Don't use region names as column headers for the feature matrix")
    print("  • The columns represent connections TO other regions, not the regions themselves")
    print("  • Use generic column names like 'conn_0', 'conn_1', etc.")
    print("  • Or create proper connection names (but it's complex due to diagonal removal)")
    
    # ============================================
    # STEP 5: Show the correct way
    # ============================================
    print("\n✅ STEP 5: The correct approach")
    print("-" * 40)
    
    print("\nFor each brain region sample:")
    print("  • The LABEL (y) tells us which region it is (0 to {})".format(n_regions-1))
    print("  • The FEATURES (X) are its {} connections to OTHER regions".format(n_regions-1))
    print("  • The MODEL learns: 'Based on your connections, you must be region X'")
    
    print("\n🎯 It's like a detective game:")
    print("  • Detective (model) sees someone's friend network (connections)")
    print("  • Detective guesses who the person is (region identity)")
    print("  • If the person acts differently (task), detective makes mistakes")
    print("  • Those mistakes show us which regions changed during the task!")
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n" + "="*70)
    print("DEBUGGING COMPLETE")
    print("="*70)
    
    print(f"""
    📊 Your Data Summary:
    • Subjects: {df_piop2.shape[0]}
    • Brain Regions: {n_regions}
    • Total Connections: {len(connection_columns)}
    • Features per Region: {n_regions-1} (after removing diagonal)
    • Total Training Samples: {df_piop2.shape[0] * n_regions}
    
    💡 Key Insight:
    The number of features ({n_regions-1}) ≠ number of regions ({n_regions})
    because we remove self-connections!
    
    🚀 Next Steps:
    1. Use the fixed train_model_fixed.py script
    2. Run with: python scripts/train_model_fixed.py --fold 1
    3. Check the outputs in data/processed/ and data/results/
    """)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
