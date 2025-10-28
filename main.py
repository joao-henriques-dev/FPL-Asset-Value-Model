from src.system import System

if __name__ == "__main__":

    gw_number = System.get_gw_number()

    system = System(gw_number)

    system.run()

    system.get_fdr_data()

    system.get_assets_value(show_gk=True, show_def=True, show_mid=True, show_forward=True)

    #System.plot_training_data()