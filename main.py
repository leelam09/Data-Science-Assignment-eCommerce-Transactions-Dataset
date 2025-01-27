from leelam_yadu_eda import perform_eda
from leelam_yadu_lookalike_model import build_lookalike_model
from leelam_yadu_clustering import perform_clustering

if __name__ == '__main__':
    print("Starting project...")
    
    # Create output folder
    import os
    if not os.path.exists('./output'):
        os.makedirs('./output')

    # Run each step
    print("Step 1: Performing EDA...")
    perform_eda()

    print("Step 2: Building Lookalike Model...")
    build_lookalike_model()

    print("Step 3: Performing Clustering...")
    perform_clustering()

    print("All steps completed successfully!")
