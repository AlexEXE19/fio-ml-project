from src.utils.cli_utils import *
import time

def run_cli():
    choice = welcome()

    while choice.lower() != "q":
        if choice == "1":
            get_regression_setting()
            choice = welcome()
        elif choice == "2":
            plot_timeseries()
            choice = welcome()
        elif choice == "3":
            execute_xgb_model()
            choice = welcome()
        else:
            print("\nWrong input, try again!\n")
            time.sleep(1)
            choice = welcome()
