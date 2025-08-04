import os
import sys

# Ensure project root is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import eda

if __name__ == "__main__":
    print("[INFO] Starting EDA process...")
    eda.main()
    print("[INFO] EDA completed. Plots saved in results/plots/eda/")

