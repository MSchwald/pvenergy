import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from dataanalysis import Pipeline, ML_MODELS, Model
from feature_catalog import FeatureCatalog as F

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PLOT_DIR = BASE_DIR / "static" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 16

class Plot:
    def weather_forecast(
            df: pd.DataFrame,
            system_id: int,
            show_title: bool = False
        ) -> Path:
        """Creates three plots of features from OpenMeteo weather forecast"""
        fig, axes = plt.subplots(1, 3, figsize=(18,6), sharex=True)
        if show_title:
            fig.suptitle(f"Weather Forecast System {system_id}, starting from {df.index[0]} local time", fontsize=16)

        # Solar radiation
        ax1 = axes[0]
        ax1.plot(df.index, df[F.GHI.name], label=F.GHI.display_name)
        ax1.plot(df.index, df[F.DNI.name], label=F.DNI.display_name)
        ax1.plot(df.index, df[F.DHI.name], label=F.DHI.display_name)
        ax1.set_ylabel("Radiation [W/m²]")
        ax1.set_ylim(bottom=0)
        ax1.legend()

        # Temperature
        ax2 = axes[1]
        ax2.plot(df.index, df[F.AIR_TEMP.name], color='tab:orange', label=F.AIR_TEMP.display_name)
        ax2.set_ylabel(f"Temperature [{F.AIR_TEMP.unit}]")
        #ax2.set_ylim(df[F.AIR_TEMP.name].min() - 2, df[F.AIR_TEMP.name].max() + 2)
        ax2.legend()

        # Wind (Speed und Direction)
        ax3 = axes[2]
        ax3.plot(df.index, df[F.WIND_SPEED.name], color='tab:blue', label=F.WIND_SPEED.display_name)
        ax3.set_ylabel(F.WIND_SPEED.display_name_with_unit, color='tab:blue')
        ax3.tick_params(axis='y', labelcolor='tab:blue')

        ax_wind_dir = ax3.twinx()
        ax_wind_dir.plot(df.index, df[F.WIND_DIRECTION.name], color='tab:green', label=F.WIND_DIRECTION.display_name)
        ax_wind_dir.set_ylabel(F.WIND_DIRECTION.display_name_with_unit, color='tab:green')
        ax_wind_dir.tick_params(axis='y', labelcolor='tab:green')

        # X axis (time)
        for ax in axes:
            ax.set_xlim(df.index[0], df.index[-1])
            ax.set_xticks(df.index)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        file = PLOT_DIR / f"forecast_{system_id}.png"
        fig.savefig(file)

        plt.close(fig)

        return file
    
    def calculated_features(
            df: pd.DataFrame,
            system_id: int,
            show_title: bool = False
        ) -> Path:
        """Creates three plots of some intuitive features calculated from the weather forecast"""
        fig, axes = plt.subplots(1, 3, figsize=(18,6), sharex=True)
        if show_title:
            fig.suptitle(f"Some Calculated Features For System {system_id}, starting from {df.index[0]} local time", fontsize=16)

        # Solar radiation and geometry
        ax1 = axes[0]
        ax1.plot(df.index, df.ftr.get(F.PVLIB_POA_IRRADIANCE), color='tab:orange', label=F.PVLIB_POA_IRRADIANCE.display_name)
        ax1.set_ylabel(F.PVLIB_POA_IRRADIANCE.display_name_with_unit, color='tab:orange')
        ax1.tick_params(axis='y', labelcolor='tab:orange')
        
        ax_angle = ax1.twinx()
        ax_angle.plot(df.index, df.ftr.get(F.AOI), color='tab:blue', linestyle=':', label=F.AOI.display_name)
        ax_angle.plot(df.index, df.ftr.get(F.RELATIVE_AZIMUTH), color='tab:blue', linestyle='--', label=F.RELATIVE_AZIMUTH.display_name)
        ax_angle.set_ylabel("Angle [°]", color='tab:blue')
        ax_angle.tick_params(axis='y', labelcolor='tab:blue')
        ax_angle.legend()

        # Cloudiness measurements
        ax2 = axes[1]
        ax2.plot(df.index, df.ftr.get(F.DHI_PER_GHI), label=F.DHI_PER_GHI.display_name)
        ax2.plot(df.index, df.ftr.get(F.CLEAR_SKY_RATIO), label=F.CLEAR_SKY_RATIO.display_name)
        ax2.set_ylabel(f"Ratio")
        ax2.set_ylim(bottom = 0, top = 1)
        ax2.legend()

        # Cooling and therming effects
        ax3 = axes[2]
        ax3.plot(df.index, df.ftr.get(F.WIND_NORMAL_COMPONENT), color='tab:blue', label=F.WIND_NORMAL_COMPONENT.display_name)
        ax3.set_ylabel(F.WIND_NORMAL_COMPONENT.display_name_with_unit, color='tab:blue')
        ax3.tick_params(axis='y', labelcolor='tab:blue')

        ax_temp = ax3.twinx()
        ax_temp.plot(df.index, df.ftr.get(F.GAMMA_TEMP_DIFFERENCE), color='tab:green', label=F.GAMMA_TEMP_DIFFERENCE.display_name_with_unit)
        ax_temp.set_ylabel(F.GAMMA_TEMP_DIFFERENCE.display_name_with_unit, color='tab:green')
        ax_temp.tick_params(axis='y', labelcolor='tab:green')

        # X axis (time)
        for ax in axes:
            ax.set_xlim(df.index[0], df.index[-1])
            ax.set_xticks(df.index)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        file = PLOT_DIR / f"calculated_features_{system_id}.png"
        fig.savefig(file)

        plt.close(fig)

        return file
    
    def predict(
            Y: pd.DataFrame,
            system_id: int,
            show_title: bool = False
        ) -> Path:
        """Creates three plots of dcp prediction for all three trained models"""
        fig, axes = plt.subplots(1, 3, figsize=(18,6), sharex=True)
        if show_title:
            fig.suptitle(f"DC Power prediction for System {system_id}, starting from {Y.index[0]} local time", fontsize=16)

        for i in range(3):
            y = Y.iloc[:,i]
            axes[i].plot(Y.index, y, color='tab:orange', label=y.name)
            axes[i].set_ylabel("DC Power [W]")
            axes[i].legend()

        # X axis (time)
        for ax in axes:
            ax.set_xlim(Y.index[0], Y.index[-1])
            ax.set_xticks(Y.index)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        file = PLOT_DIR / f"dcp_prediction_{system_id}.png"
        fig.savefig(file)

        plt.close(fig)

        return file
    
if __name__ == "__main__":
    """Testing space for designing plots"""
    #m = Model.load("xgboost_all_ids")
    df = Pipeline.weather_forecast(2)
    ml_models = [Model.load(ml_model.name) for ml_model in [ML_MODELS.XGBOOST, ML_MODELS.LIGHTGBM, ML_MODELS.RANDOM_FOREST]]
    Y = Pipeline.predict(ml_models, df)
    Plot.predict(Y, 2)
    #print(Pipeline.predict(ml_models, df))