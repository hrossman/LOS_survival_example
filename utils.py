import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times



def load_4th_wave_data():
    moh_df = pd.read_excel('moh_data.xlsx', skiprows=1, parse_dates=['תאריך'])
    moh_df.columns = ['date', 'new_severe']
    new_hosped = moh_df.iloc[140:301]['new_severe'].values
    times = np.arange(len(new_hosped))
    return times, new_hosped


def plot_new_hosped(times, new_hosped, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(times, new_hosped, color='k', lw=3, alpha=0.8, label='New severe patients')
    ax.set_ylabel('New severe patients')
    ax.set_xlabel('Day of estimation')
    ax.grid()


def log_normal_int_dist(ln_mean:float, ln_std:float, size:int, round:bool=False, max_val:int=30) -> np.ndarray:
    """Just a function to sample lognormal dist
    """
    mean = np.log(
        ln_mean ** 2 / np.sqrt(ln_std ** 2 + ln_mean ** 2)
    )  # Computes the mean of the underlying normal distribution
    sigma = np.sqrt(
        np.log(ln_std ** 2 / ln_mean ** 2 + 1)
    )  # Computes sigma for the underlying normal distribution
    samples = np.random.lognormal(mean=mean, sigma=sigma, size=size)
    samples = np.clip(samples, a_min=0, a_max=max_val)
    if round:
        # round to integer
        samples = np.round(samples)
    return samples


def plot_lognormal_dist(ln_mean, ln_std, max_val, size=100_000):
    los = log_normal_int_dist(ln_mean=ln_mean, ln_std=ln_std, size=size, round=False, max_val=max_val)
    fig, ax = plt.subplots(1,1)
    sns.kdeplot(los, ax=ax)
    ax.axvline(np.median(los), color='r')
    ax.set_xlim(0,50)
    true_median = np.median(los)
    ax.text(true_median+0.5, 0.01, f"{true_median:.2f}", color='r')
    return true_median


def generate_los_data(ln_mean, ln_std, max_val, new_hosped):
    los = []
    t_adm = []
    t_release = []

    for t, num_new_hosped in enumerate(new_hosped):
        tmp_t_adm = np.full(num_new_hosped, t)
        t_adm.extend(tmp_t_adm)
        tmp_los = log_normal_int_dist(ln_mean=ln_mean, ln_std=ln_std, size=num_new_hosped, round=False, max_val=max_val)
        los.extend(tmp_los)
        
    data = pd.DataFrame({'t_adm': t_adm, 'los': los})
    data['t_release'] = data['t_adm'] + data['los']
    return data


# def animate():
#     from celluloid import Camera

#     fig, axes = plt.subplots(1,2, figsize=(12,5))
#     camera = Camera(fig)

#     ax = axes[0]
#     ax.plot(times[START_DAY:], new_hosped[START_DAY:], color='k', lw=3, alpha=0.8, label='New severe patients')
#     ax.set_ylabel('New severe patients', fontsize=sz)
#     ax.set_xlabel('Day of estimation', fontsize=sz)

#     ax = axes[1]
#     sns.kdeplot(data['los'], ax=ax)
#     ax.axvline(np.median(los), color='r', lw=5, alpha=0.6)
#     ax.set_xlim(0,15)

#     fig.tight_layout()
#     fig.patch.set_facecolor('white')

#     for current_t in times[START_DAY:]:
#         ax = axes[0]
#         ax.plot(times[START_DAY:], new_hosped[START_DAY:], color='k', lw=3, alpha=0.8, label='New severe patients')
#         ax.fill_between(x=np.arange(START_DAY, current_t), y1=np.zeros_like(new_hosped[START_DAY:current_t]), y2=new_hosped[START_DAY:current_t], color='gray')


#         ax = axes[1]
#         # sns.kdeplot(data['los'], ax=ax, color='r', alpha=0.6)
#         ax.axvline(np.median(los), color='r', lw=6, alpha=0.5)

#         ax.axvline(km_median_los_7day[current_t], color='C0', lw=2, ls='-', alpha=0.9)
#         ax.axvline(naive_median_los_7day[current_t], color='C1', lw=2, ls='-', alpha=0.9)
#         camera.snap()


#     animation = camera.animate()
#     animation.save('animation.gif')