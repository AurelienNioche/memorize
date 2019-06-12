#
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from os import path
#
# import multiprocessing as MP
# import sys
# from collections import defaultdict
import os
from scipy.stats import mannwhitneyu
import datetime
import dill

from spaced_rep_code import *
from plot_utils import *
# get_unique_user_lexeme, get_training_pairs, calc_user_LL_dict, \
#    calc_LL_dict_threshold, merge_with_thres_LL, get_all_durations,


from backup.backup import save, load

FIG_FOLDER = "fig"
os.makedirs(FIG_FOLDER, exist_ok=True)

# Data files should be kept at these paths.
RAW_DATA = "./data/duolingo_reduced.csv"
DICT_DATA = "./data/duo_dict.dill"
MODEL_WEIGHTS_FILE = "power.duolingo.weights"

FILE_RESULTS_DUO = os.path.join("data", "results_duo.p")
FILE_DUO_PAIRS = os.path.join("data", "duo_pairs.p")
FILE_DF_DUO = os.path.join("data", "df_duo.p")
FILE_TRAINING_PAIRS = os.path.join("data", "training_pairs.p")
FILE_DUO_DICT = os.path.join("data", "duo_dict.p")

sns.set_style("ticks")
sns.set_context("paper", font_scale=2.5, rc={'lines.linewidth': 3,
                                             'lines.markersize': 10,
                                             'legend.fontsize': 24})
sns.set_palette("Set1")


class AnalysisObjects:

    def __init__(self):
        self._duo_pairs = None
        self._duo_dict = None
        self._training_pairs = None
        self._initial_cond = None
        self._df_duo = None
        self._df_duo_index_set = None

    @property
    def duo_pairs(self):

        if self._duo_pairs is None:
            self._duo_pairs = \
                self.load_if_not_existing(file_path=FILE_DUO_PAIRS,
                                          method=get_unique_user_lexeme,
                                          duo_dict=self.duo_dict)
        return self._duo_pairs

    def totals(self, ):

        total_correct = []
        total_seen = []
        for ind, (u_id, l_id) in enumerate(self.duo_pairs):
            for item in self.duo_dict[u_id][l_id]:
                time = self.initial_cond[(l_id, u_id)]
                correct = self.df_duo_index_set["history_correct"].loc[(l_id, u_id, time)].tolist()[0]
                seen = self.df_duo_index_set["history_seen"].loc[(l_id, u_id, time)].tolist()[0]
                total_correct.append(correct)
                total_seen.append(seen)


    # def right_wrong(self):
    #
    #     results_duo, duo_pairs, initial_cond, duo_dict, df_duo_index_set, duo_lexeme_difficulty,
    #     duo_map_lexeme
    #
    #     right = results_duo.loc['right'][0]
    #     wrong = results_duo.loc['wrong'][0]
    #     temp_duo_map_lexeme = {}
    #     for ind, (u_id, l_id) in enumerate(duo_pairs):
    #         if ind % 1000 == 0:
    #             print(datetime.datetime.now().isoformat(), ind, '/', len(duo_pairs))
    #         for item in duo_dict[u_id][l_id]:
    #             time = initial_cond[(l_id, u_id)]
    #             correct = df_duo_index_set["history_correct"].loc[(l_id, u_id, time)].tolist()[0]
    #             seen = df_duo_index_set["history_seen"].loc[(l_id, u_id, time)].tolist()[0]
    #             # print(correct, seen)
    #             temp = None
    #             if l_id not in temp_duo_map_lexeme:
    #                 temp_duo_map_lexeme[l_id] = duo_lexeme_difficulty[duo_map_lexeme[l_id]]
    #             temp = temp_duo_map_lexeme[l_id]
    #             item['n_0'] = temp * 2 ** (-(right * correct + wrong * (seen - correct)))



    @property
    def training_pairs(self):

        if self._training_pairs is None:
            self._training_pairs = self.load_if_not_existing(file_path=FILE_TRAINING_PAIRS, method=get_training_pairs,
                                                             data_dict=self.duo_dict, duo_pairs=self.duo_pairs)

        print(f'{len(self._training_pairs) / len(self.duo_pairs) * 100.:.2f} of sequences can be used for training/testing.')
        return self._training_pairs

    @property
    def initial_cond(self):

        if self._initial_cond is None:
            self._initial_cond = self.load_if_not_existing(file_path=)

    @property
    def duo_dict(self):

        if self._duo_dict is None:
            self._duo_dict = self.load_if_not_existing(
                file_path=FILE_DUO_DICT,
                method=self.load_duo_dict)

        return self._duo_dict

    @property
    def df_duo(self):

        if self._df_duo is None:
            self._df_duo = self.load_if_not_existing(
                file_name='df_duo.p',
                method=self.load_df_duo
            )

        return self._df_duo

    @property
    def df_duo_index_set(self):

        if self._df_duo_index_set is None:
            self._df_duo_index_set = self.load_if_not_existing(
                file_name=""
            )

    @staticmethod
    def load_if_not_existing(file_name, method, **kwargs):

        file_path = os.path.join("data", file_name)

        if os.path.exists(file_path):
            data = load(file_path)

        else:
            data = method(**kwargs)
            save(obj=data, file_name=file_path)

        return data

    @staticmethod
    def load_results_duo():

        print('Loading weights...', end=" ", flush=True)
        results_duo = pd.read_csv(open(MODEL_WEIGHTS_FILE, 'rb'),
                                  sep="\t",
                                  names=['label', 'value'],
                                  header=None)

        results_duo.set_index("label", inplace=True)
        print("Done!")
        return results_duo

    @staticmethod
    def load_df_duo():

        # Load Data
        print('Loading raw data...', end=" ", flush=True)
        df_duo = pd.read_csv(RAW_DATA)
        print("Done!")

        df_duo['lexeme_comp'] = df_duo['learning_language'] + ":" + df_duo['lexeme_string']

        # df_duo.set_index(["lexeme_id", "user_id", "timestamp"], inplace=True)
        # df_duo.sort_index(inplace=True)

        return df_duo

    def create_initial_cond(self):

        initial_cond = self.df_duo[["lexeme_id", "user_id", "timestamp"]].groupby(["lexeme_id", "user_id"]).min()[
            "timestamp"]
        initial_cond = initial_cond.to_dict()
        return initial_cond


    def create_df_duo_index_set(self):
        df_duo_index_set = self.df_duo.copy().set_index(
            ["lexeme_id", "user_id", "timestamp"])
        df_duo_index_set.sort_index(inplace=True)
        return df_duo_index_set

def main():
    pass
    # # Load Data
    # df_duo = load_if_not_existing(FILE_DF_DUO, method=load_df_duo)
    # results_duo = load_if_not_existing(FILE_RESULTS_DUO, method=load_results_duo)
    #
    # print("convert")
    # convert = df_duo[['lexeme_comp', 'lexeme_id']].set_index('lexeme_comp').to_dict()['lexeme_id']
    #
    # # Load Data
    #
    # # B = results_duo.loc["B"]
    # start = 5
    #
    # duo_lexeme_difficulty = 2 ** (-(results_duo[start:] + results_duo.loc['bias']))
    # new_index = []
    # for ind in duo_lexeme_difficulty.index:
    #     new_index.append(convert[ind])
    #
    # duo_lexeme_difficulty['new_index'] = new_index
    # duo_lexeme_difficulty.set_index('new_index', inplace=True)
    #
    # duo_map_lexeme = dict([(l_id, ind) for ind, l_id in enumerate(duo_lexeme_difficulty.index)])
    #
    # duo_lexeme_difficulty = duo_lexeme_difficulty['value'].tolist()
    # duo_alpha = (-2 ** (-results_duo.loc['right']) + 1).iloc[0]
    # duo_beta = (2 ** (-results_duo.loc['wrong']) - 1).iloc[0]
    #
    # print('Loading duo dictionary...', end=" ", flush=True)
    # duo_dict = dill.load(open(dict_data, 'rb'))
    # # len(duo_dict)
    # print('Done!')
    #
    # print('Doing some preprocessing...', end=' ', flush=True)
    # duo_pairs = get_unique_user_lexeme(duo_dict)
    #
    # initial_cond = df_duo[["lexeme_id", "user_id", "timestamp"]].groupby(["lexeme_id", "user_id"]).min()["timestamp"]
    # initial_cond = initial_cond.to_dict()
    # # df_duo_index_set = df_duo.copy().set_index(["lexeme_id", "user_id", "timestamp"])
    # df_duo_index_set = df_duo.copy().set_index(["lexeme_id", "user_id", "timestamp"])
    # df_duo_index_set.sort_index(inplace=True)
    # print('Done!')
    #
    # duo_stats_99 = calc_user_LL_dict(duo_dict, duo_alpha, duo_beta, duo_lexeme_difficulty, duo_map_lexeme,
    #                                  success_prob=0.99, pairs=duo_pairs, verbose=False)
    #
    # with warnings.catch_warnings():
    #     warnings.filterwarnings('error')
    #     duo_stats_training = calc_user_LL_dict(
    #         duo_dict, duo_alpha, duo_beta, duo_lexeme_difficulty, duo_map_lexeme,
    #         success_prob=0.99, training=True, pairs=training_pairs, verbose=False,
    #         n_procs=None
    #     )
    #
    # threshold_LL = calc_LL_dict_threshold(duo_dict, alpha=duo_alpha, beta=duo_beta, pairs=duo_pairs,
    #                                       lexeme_difficulty=duo_lexeme_difficulty, map_lexeme=duo_map_lexeme,
    #                                       success_prob=0.99, verbose=False)
    # merge_with_thres_LL(duo_stats_99, threshold_LL, pairs=duo_pairs)
    #
    # threshold_LL_training = calc_user_LL_dict_threshold(
    #     duo_dict, alpha=duo_alpha, beta=duo_beta, pairs=training_pairs,
    #     lexeme_difficulty=duo_lexeme_difficulty, map_lexeme=duo_map_lexeme,
    #     success_prob=0.99, verbose=False, training=True)
    #
    # merge_with_thres_LL(duo_stats_training, threshold_LL_training, pairs=training_pairs)
    #
    # duo_durations = get_all_durations(duo_dict, duo_pairs)
    #
    # # Different sequences be chosen for different T
    # # The paper contains plots correspondnig to T \in {3, 5, 7}
    # middle_dur_pairs = filter_by_duration(
    #     durations_dict=duo_durations,
    #     pairs=training_pairs,
    #     T=3, alpha=0.1,
    #     verbose=True)
    #
    # # Calculate the metric
    # duo_forgetting_rate = calc_empirical_forgetting_rate(duo_dict, pairs=duo_pairs, no_norm=False)
    # base = calc_empirical_forgetting_rate(duo_dict, pairs=duo_pairs, return_base=True)
    #
    # perf = duo_forgetting_rate
    # with_exact_reps = True
    # quantile = 0.25
    # with_training = True
    #
    #
    # def top_k_reps_worker(reps):
    #     max_reps = None if not with_exact_reps else reps + 1
    #     stats_dict = duo_stats_99 if not with_training else duo_stats_training
    #     # pairs = duo_pairs if not with_training else training_pairs
    #     pairs = duo_pairs if not with_training else middle_dur_pairs
    #     return reps, calc_top_k_perf(stats_dict, perf, pairs=pairs, quantile=quantile,
    #                                  min_reps=reps, max_reps=max_reps, with_overall=True,
    #                                  only_finite=False, with_threshold=True)
    #
    #
    # reps_range = np.arange(1 if not with_training else 2, 8)
    #
    # # For performance
    # with MP.Pool(9) as pool:
    #     top_k_99_reps = pool.map(top_k_reps_worker, reps_range)
    #
    # # For debugging
    # # top_k_99_reps = []
    # # for i in reps_range:
    # #    top_k_99_reps.append(top_k_reps_worker(i))
    #
    # stats = {}
    # for i in range(len(top_k_99_reps)):
    #     mem_thresh = mannwhitneyu(top_k_99_reps[i][1]['perf_top_threshold'],
    #                               top_k_99_reps[i][1]['perf_top_mem'])
    #     mem_unif = mannwhitneyu(
    #         top_k_99_reps[i][1]['perf_top_mem'], top_k_99_reps[i][1]['perf_top_unif'])
    #     stats[top_k_99_reps[i][0]] = (mem_thresh, mem_unif)
    #     print(top_k_99_reps[i][0], "MEM vs. Threshold", stats[top_k_99_reps[i][0]][0])
    #     print(top_k_99_reps[i][0], "MEM vs. Uniform", stats[top_k_99_reps[i][0]][1])
    #
    # # T=7.0 alpha=0.1
    # # latexify(fig_width=2,largeFonts=False,columns=2,font_scale=1.0)
    # latexify(fig_width=3.4, largeFonts=True)
    #
    # plot_perf_by_reps_boxed(top_k_99_reps, with_threshold=True, std=False,
    #                         max_rev=7, median=True, stats=stats)
    #
    # format_axes(plt.gca())
    # plt.ylabel("$\hat{n}$")
    # plt.xlabel("\# reviews")
    #
    # plt.ylim(0, 0.4)
    # plt.savefig(os.path.join(FIG_FOLDER, 'empirical_p_recall_duo_new_boxed_split_T_3.pdf'), bbox_inches='tight',
    #             pad_inches=0)
    # plt.savefig(os.path.join(FIG_FOLDER, 'empirical_p_recall_duo_new_boxed.pdf'), bbox_inches='tight', pad_inches=0)
    # # plt.xlim(0,6*2+)
    #
    # middle_dur_pairs = filter_by_duration(
    #     durations_dict=duo_durations,
    #     pairs=training_pairs,
    #     T=8, alpha=0.4,
    #     verbose=True)
    #
    # users = list(map(lambda f: f[0], middle_dur_pairs))
    #
    # from collections import OrderedDict
    #
    # users_ = {}
    # for u in users:
    #     if u in users_:
    #         users_[u] += 1
    #     else:
    #         users_[u] = 0
    # users_ = OrderedDict(sorted(users_.items(), key=lambda x: -x[1]))
    #
    # users_ = OrderedDict(filter(lambda x: x[1] > 70, users_.items()))
    # print('Number of users = ', len(users_))
    #
    # results = []
    # for ind, u in enumerate(users_):
    #     if ind % 100 == 0:
    #         print("Completed = ", ind / len(users_))
    #
    #     middle_dur_pairs_users = filter_by_users(middle_dur_pairs, [u], False)
    #     perf = duo_forgetting_rate
    #     with_exact_reps = True
    #     with_training = True
    #     threshold = 1.0
    #
    #     def top_k_reps_worker(reps):
    #         max_reps = None if not with_exact_reps else reps + 1
    #         stats_dict = duo_stats_99 if not with_training else duo_stats_training
    #         # pairs = duo_pairs if not with_training else training_pairs
    #         pairs = middle_dur_pairs_users
    #         return reps, calc_top_memorize(stats_dict, perf, pairs=pairs,
    #                                        min_reps=reps, max_reps=max_reps, with_overall=True,
    #                                        only_finite=True, with_threshold=True)
    #
    #     reps_range = np.arange(1 if not with_training else 2, 10)
    #
    #     # For performance
    #     # with MP.Pool(9) as pool:
    #     #    top_k_99_reps = pool.map(top_k_reps_worker, reps_range)
    #     #
    #     # For debugging
    #     # top_k_99_reps = []
    #     for i in reps_range:
    #         reps, (corr_mem, corr_uniform, corr_thresh) = top_k_reps_worker(i)
    #         if corr_mem[1] < threshold:
    #             results.append((u, reps, corr_mem[0], "Memorize"))
    #
    # # c1, c2, c3 = sns.color_palette("Set2", n_colors=3)
    # df = pd.DataFrame(results, columns=["user", "Repetitions",
    #                                     "Pearson Correlation\nCoefficient", "Policy"])
    # # df.plot("reps","correlation",kind="bar")
    # df = df[df['Repetitions'] < 8]
    # df = df[df["Policy"] == "Memorize"]
    # sns.factorplot("Repetitions", "Pearson Correlation\nCoefficient", hue="Policy", data=df, size=2.2,
    #                aspect=2, legend=False, estimator=np.median, ci=68, capsize=.2,
    #                # ,hue_order=["Memorize", "Threshold","Uniform"],
    #                palette=sns.color_palette("Set2", n_colors=3), linestyles=[""])
    # plt.hlines(0, -1, 7, linestyle="--")
    # plt.legend(loc=9, ncol=3)
    # plt.savefig(os.path.join(FIG_FOLDER, 'user_n_perf_all.pdf'), bbox_inches='tight',pad_inches=0)


if __name__ == "__main__":

    main()
