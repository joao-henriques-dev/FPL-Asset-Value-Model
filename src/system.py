from utils import data_loader
from src.asset_value_model import AssetValueModel
from src.fdr_value_model import FDRValueModel
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class System:
    """
    Represents the system responsible for handling models and data related to fixture difficulty
    rating (FDR) values and asset values.

    This class integrates various subsystems and data models to provide simplified interfaces
    for retrieving FDR values and asset values. Additionally, it includes methods for loading
    watchlist data of players categorised by their roles in the game and for visualising training
    data.

    Attributes:
        fdr_value_model (FDRValueModel): Instance of the FDRValueModel class, initialised with
            fixture data, that computes FDR values for all PL teams
        assets_value_model (AssetValueModel): Instance of the AssetValueModel class,
            initialised with watchlist data, fixture data, and specific pre-processed
            training data (xGCp90 and defensive points per match), that computes
            asset values for goalkeepers, defenders, midfielders, and forwards
        gw (int): Represents the current game week for the system's context
        wc_gw (int): The game week number when the user will use their Wildcard
        fh_gw (int): The game week number when the user will use their Free Hit
        is_running (bool): Indicates whether the system is running or not
    """

    def __init__(self, gw):
        """
        Initialises the System object with the given game week number.

        Args:
            gw (int): The current game week number
        """
        self.fdr_value_model = FDRValueModel(data_loader.load_fixtures_data())

        training_data = data_loader.load_training_data()

        xgcp90_data = training_data["xGCp90"]
        defensive_points_data = training_data["defensive points p/match"]
        xgcp90_data = xgcp90_data.to_numpy()
        defensive_points_data = defensive_points_data.to_numpy()
        clean_sheets_per_match = training_data["clean sheets p/match"]
        clean_sheets_per_match = clean_sheets_per_match.to_numpy()

        self.assets_value_model = AssetValueModel(self.get_watchlist_data(),
            data_loader.load_fixtures_data(), xgcp90_data,
            defensive_points_data, clean_sheets_per_match)

        self.gw = gw
        self.wc_gw = 0
        self.fh_gw = 0
        self.is_running = False

    def run(self) -> None:
        """
        Runs the system by welcoming the user to the app and then calling a method to prompt
        them for the game weeks in which they plan to use their Wildcard and Free Hit chips.
        """
        print("""
Greetings!
Welcome to the FPL Assets Value Model!
A revolutionary tool for calculating the value of FPL assets, helping you
make informed and data-driven decisions to help you achieve your rank goals.

So far, we provide the following functionality:
    - custom Fixture Difficulty Ratings (FDR) plots for Premier League teams;
    - custom value plots for popular players of all positions:
        - goalkeepers;
        - defenders;
        - midfielders;
        - forwards.

Before you jump into the results, and to enhance your experience, we need to know
how you plan to approach this FPL season. Practically speaking, when do you plan to
use your Wildcard and Free Hit, if you still have them?
It's completely fine, in the case where you haven't used them, if you don't know
yet when to do so, but, in the case you already have a GW in mind to use these chips,
know that this decision directly affects the value of your assets and the best teams.
            
Wish you an amazing FPL season, or the rest of it if you're midway through it!
            
- João Henriques
""")

        self.wc_gw = self.get_chip_gw("Wildcard")
        self.fh_gw = self.get_chip_gw("Free Hit")
        self.is_running = True

    def get_fdr_data(self) -> None:
        """
        Plots FDR data for each PL team using a bar plot.
        The data is retrieved from the FDRValueModel object.
        """
        if not self.is_running:
            print("You need to run the system first.")
            return

        fdr_palette = System.create_palette(0, 20, 1, "descending")
        df = self.fdr_value_model.get_fdr_value(self.gw, self.wc_gw, self.fh_gw)
        fig, ax = plt.subplots()
        sns.barplot(data=df, y="team", x="FDR value", hue="position", palette=fdr_palette)
        plt.title("Fixture Difficulty Ratings values")
        plt.legend(
            title='position',
            fontsize=8,
            title_fontsize=9,
            bbox_to_anchor=(1, 1), loc='upper left'
        )
        plt.xlabel('')
        plt.ylabel('')
        fig.text(
            0.99, 0.01, "Made by Vigor © 2025",
            ha='right', va='bottom', fontsize=8, color='gray'
        )
        plt.tight_layout()
        plt.show()

    def get_assets_value(self, show_gk=True, show_def=True, show_mid=True, show_forward=True) -> None:
        """
        Gets the dataframes for goalkeepers, defenders, midfielders, and forwards and
        plots their respective values using a bar plot with the specific palettes.
        The data is retrieved from the AssetValueModel object.

        Args:
            show_gk (bool): Indicates whether to show goalkeepers' value plot
            show_def (bool): Indicates whether to show defenders' value plot
            show_mid (bool): Indicates whether to show midfielders' value plot
            show_forward (bool): Indicates whether to show forwards' value plot
        """
        if not self.is_running:
            print("You need to run the system first.")
            return

        if show_gk:
            gk_palette = System.create_palette(4.0, 5.7, 0.1, "ascending")
            gk_df = self.assets_value_model.get_asset_value_gk(self.gw, self.wc_gw, self.fh_gw)
            System.plot_asset_value(gk_df, "Goalkeeper Value", gk_palette)

        if show_def:
            def_palette = System.create_palette(4.0, 6.3, 0.1, "ascending")
            def_df = self.assets_value_model.get_asset_value_def(self.gw, self.wc_gw, self.fh_gw)
            System.plot_asset_value(def_df, "Defender Value", def_palette)

        if show_mid:
            mid_palette = System.create_palette(5.0, 14.4, 0.1, "ascending")
            mid_df = self.assets_value_model.get_asset_value_mid(self.gw, self.wc_gw, self.fh_gw)
            System.plot_asset_value(mid_df, "Midfielder Value", mid_palette)

        if show_forward:
            fwd_palette = System.create_palette(5.0, 14.5, 0.1, "ascending")
            fwd_df = self.assets_value_model.get_asset_value_fwd(self.gw, self.wc_gw, self.fh_gw)
            System.plot_asset_value(fwd_df, "Forward Value", fwd_palette)

    def get_chip_gw(self, chip: str) -> int:
        """
        Prompts the user for the game week in which they plan to use a given chip.

        Args:
            chip (str): The name of the chip to be used

        Returns:
            chip_gw (int): The game week number when the user will use the chip
        """
        is_valid_gw = False
        chip_gw = 0

        while not is_valid_gw:
            try:
                chip_gw = int(input(f"""
Enter the GW in which you plan to use your {chip}.
(press '0' if you don't know yet when to use your {chip})
{chip} GW: """))
                if ((1 <= self.gw <= 19 and max(2, self.gw) <= chip_gw <= 19)
                    or (20 <= self.gw <= 38 and max(20, self.gw) <= chip_gw <= 38)
                    or chip_gw == 0):
                    is_valid_gw = True
                else:
                    print("Invalid input. Please enter a valid GW number.")
            except ValueError:
                print("Invalid input. Please enter a valid GW number.")

        return chip_gw

    @staticmethod
    def get_gw_number():
        try:
            gw_number = int(input("Enter the GW number: "))
        except ValueError:
            print("Invalid input. The number must be an integer between 1 and 38.")
            return System.get_gw_number()
        else:
            return gw_number

    @staticmethod
    def get_watchlist_data() -> tuple:
        """
        Loads watchlist data for goalkeepers, defenders, midfielders, and forwards
        through the corresponding data loader methods and returns a tuple of the
        mentioned dataframes.

        Returns:
            watchlist (tuple): A tuple containing dataframes
                for goalkeepers, defenders, midfielders, and forwards
        """
        return (data_loader.load_goalkeeper_data(), data_loader.load_defender_data(),
                data_loader.load_midfielder_data(), data_loader.load_forward_data())

    @staticmethod
    def create_palette(min_value: float, max_value: float, step: float, order: str) -> dict:
        """
        Creates a blue gradient palette dictionary with hex colour codes.

        Args:
            min_value (float): Minimum value for the palette range
            max_value (float): Maximum value for the palette range
            step (float): Step size for the palette range
            order (str): Indicates whether the palette will be ordered
                in ascending order or descending order

        Returns:
            dict: Dictionary mapping values to hex colour codes in blue gradient
        """
        values = []
        if order == "descending":
            current = max_value
            while current >= min_value:
                values.append(round(current, 1))
                current -= step
        else:
            current = min_value
            while current <= max_value:
                values.append(round(current, 1))
                current += step

        num_colors = len(values)
        if num_colors == 0:
            return {}

        palette = {}

        for i, value in enumerate(values):
            if num_colors == 1:
                # If only one value, use middle blue
                ratio = 0.5
            else:
                # Calculate ratio for gradient (0 = light, 1 = dark)
                ratio = i / (num_colors - 1)

            # Create darker blue gradient by shifting the lightness range down
            # Now ranges from medium to very dark
            lightness = 0.65 - (ratio * 0.55)  # From 0.65 (medium) to 0.05 (very dark)

            # Convert to RGB using darker blue ranges
            if lightness > 0.5:
                # Medium blues
                r = int(80 + (120 - 80) * (lightness - 0.45) / 0.15)
                g = int(140 + (180 - 140) * (lightness - 0.45) / 0.15)
                b = int(200 + (230 - 200) * (lightness - 0.45) / 0.15)
            elif lightness > 0.3:
                # Dark blues
                r = int(40 + (80 - 40) * (lightness - 0.3) / 0.15)
                g = int(90 + (140 - 90) * (lightness - 0.3) / 0.15)
                b = int(160 + (200 - 160) * (lightness - 0.3) / 0.15)
            elif lightness > 0.15:
                # Very dark blues
                r = int(20 + (40 - 20) * (lightness - 0.15) / 0.15)
                g = int(50 + (90 - 50) * (lightness - 0.15) / 0.15)
                b = int(120 + (160 - 120) * (lightness - 0.15) / 0.15)
            else:
                # Extremely dark blues (almost navy/black)
                r = int(0 + 20 * lightness / 0.15)
                g = int(5 + 45 * lightness / 0.15)
                b = int(60 + 60 * lightness / 0.15)

            # Ensure values are within valid range
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

            # Convert to hex
            hex_color = f"#{r:02X}{g:02X}{b:02X}"
            palette[value] = hex_color

        return palette

    @staticmethod
    def plot_asset_value(df: pd.DataFrame, title: str, palette: dict) -> None:
        """
        Plots asset values, using a bar plot, for each player in the given
        dataframe, whose position is indicated by the 'title' parameter. The
        colour scheme for the 'price' column is defined by the 'palette' parameter.

        Args:
            df (pd.DataFrame): Input dataframe containing the players' data to be plotted
            title (str): Title of the plot
            palette (dict): Dictionary defining the colour scheme for the 'price' column
        """
        fig, ax = plt.subplots()
        sns.barplot(data=df, y="player", x="value", hue="price", palette=palette)
        if len(df) > 30:
            ax.tick_params(axis='y', labelsize=7)
        plt.title(title)
        plt.legend(
            title='price',
            fontsize=8,
            title_fontsize=9,
            bbox_to_anchor=(1, 1), loc='upper left'
        )
        plt.xlabel('')
        plt.ylabel('')
        fig.text(
            0.99, 0.01, "Made by Vigor © 2025",
            ha='right', va='bottom', fontsize=8, color='gray'
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_training_data() -> None:
        """
        Plots the relationship between 'xGCp90' and 'defensive points p/match'
        using a scatter plot. The data is loaded from the training dataset.
        """
        df = data_loader.load_training_data()
        sns.scatterplot(data=df, x="xGCp90", y="defensive points p/match")
        plt.title("xGCp90 vs. defensive points p/match")
        plt.xlabel("xGCp90")
        plt.ylabel("defensive points p/match")
        plt.show()