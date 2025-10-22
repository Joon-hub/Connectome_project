'''
Scripts package for brain connectivity pipeline
===============================================
Contains executable scripts for running the analysis pipeline.
'''

# Import modules for programmatic access


__all__ = [
    'train_model',
    'apply_to_task', 
    'visualize_results'
]

def run_pipeline():
    '''Run the complete pipeline programmatically'''
    print("Running complete brain connectivity pipeline...")
    
    print("\n[1/3] Training model on PIOP-2...")
    train_model.main()
    
    print("\n[2/3] Applying model to PIOP-1...")
    apply_to_task.main()
    
    print("\n[3/3] Creating visualizations...")
    visualize_results.main()
    
    print("\n✅ Pipeline complete!")