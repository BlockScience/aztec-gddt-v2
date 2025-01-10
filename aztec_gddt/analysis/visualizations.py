from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fit_predict(X: pd.DataFrame,
      y: pd.Series,
      target: str,
      ax_dt: object,
      ax_rf: object,
      label: str = 'target',
      font_size: int = 12):
    """
    Fit DT and RF classifiers for summarizing the sensivity.
    """
    model = DecisionTreeClassifier(class_weight='balanced',
                                   max_depth=4,
                                   )
    rf = RandomForestClassifier()
    model.fit(X, y)
    rf.fit(X, y)


    df = (pd.DataFrame(list(zip(X.columns, rf.feature_importances_)),
                        columns=['features', 'importance'])
            .sort_values(by='importance', ascending=False)
            )

    plot_tree(model,
                rounded=True,
                proportion=True,
                fontsize=font_size,
                feature_names=list(X.columns),
                class_names=['threshold not met', 'threshold met'],
                filled=True,
                ax=ax_dt)
    ax_dt.set_title(
        f'Decision tree for {label}, score: {model.score(X, y) :.0%}. Trajectories: {len(X) :,}')
    sns.barplot(data=df,
                x=df.features,
                y=df.importance,
                ax=ax_rf,
                label='small')
    plt.setp(ax_rf.xaxis.get_majorticklabels(), rotation=45)
    ax_rf.tick_params(axis='x', labelsize=14)
    ax_rf.set_xlabel("Parameters", fontsize=14)
    ax_rf.set_title(f'Feature importance for the {label}')
    
    return df.assign(target=target)



def param_sensitivity_plot(df: pd.DataFrame,
                           control_params: set,
                           target: str,
                           label: str = 'target',
                           height: int = 12,
                           width: int = 30,
                           font_size: int = 8,
                           axes = None):
    """
    Plot the sensivity of the 'target' column vs
    a list of control parameters, which are data frame columns.
    """
    
    features = set(control_params) - {target}
    X = df.loc[:, list(features)]
    y = (df[target] > 0)
    # Visualize
    if axes is None:
        fig, axes = plt.subplots(nrows=2,
                                figsize=(width, height),
                                dpi=72,
                                gridspec_kw={'height_ratios': [3, 1]})
    fit_predict(X, y, 'target', axes[0], axes[1], label, font_size)

    return None


def plot_agg_kpis(agg_df, control_params, AggKPIs):
    N_cols = len(AggKPIs)

    fig, axes = plt.subplots(nrows=2,
                            ncols=N_cols,
                            figsize=(30, 12),
                            dpi=72,
                            gridspec_kw={'height_ratios': [3, 1]})

    for i, aggKPI in enumerate(AggKPIs):
        i_axes = (axes[0][i], axes[1][i])
        param_sensitivity_plot(agg_df, control_params, aggKPI, label=f'{aggKPI}', axes=i_axes)



def plot_inspect_vars(sim_df):
    from random import choices
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    cols_1 = ['cumm_total_tx', 'cumm_dropped_tx', 'cumm_excl_tx'] # Count

    cols_2 = ['cumm_finalized_epochs', 'cumm_resolved_epochs', 'cumm_unproven_epochs'] # Count

    cols_3 = ['cumm_finalized_blocks', 'cumm_empty_blocks'] # Count

    cols_4 = ['market_price_juice_per_wei', 'oracle_price_juice_per_wei', ] # Juice per Wei

    cols_5 = ['market_price_l1_gas', 'market_price_l1_blobgas', 'oracle_price_l1_gas', 'oracle_price_l1_blobgas'] # Wei per Gas

    cols_6 = ['base_fee'] # Juice per Mana

    cols_7 = ['oracle_proving_cost'] # Wei per Mana


    N_subsets = 5
    N_cols = 7
    size_per_col = 6
    size_per_row = 3

    groupings = list(sim_df.groupby(['simulation', 'subset', 'run',]))
    trajectory_dfs = choices([el[1] for el in groupings], k=N_subsets)

    X_COL = 'l1_blocks_passed'
        
    fig, axes = plt.subplots(nrows=N_subsets, ncols=N_cols, figsize=(N_cols*size_per_col, N_subsets*size_per_row), sharex=True)
    fig.tight_layout()
    for i in range(N_subsets):
        traj_df = (trajectory_dfs[i])

        melted_df = traj_df.reset_index().melt(id_vars=[X_COL], value_vars=cols_1)
        ax=axes[i][0]
        sns.lineplot(melted_df, x=X_COL, y='value', hue='variable', ax=ax)
        ax.grid()
        ax.set_yscale('linear')
        ax.set_ylabel('Count')
        if i < N_subsets - 1:
           ax.get_legend().remove()
        
        ax = axes[i][1]
        melted_df = traj_df.reset_index().melt(id_vars=[X_COL], value_vars=cols_2)
        sns.lineplot(melted_df, x=X_COL, y='value', hue='variable', ax=ax)
        ax.grid()
        ax.set_yscale('linear')
        ax.set_ylabel('Count')
        if i < N_subsets - 1:
         ax.get_legend().remove()


        melted_df = traj_df.reset_index().melt(id_vars=[X_COL], value_vars=cols_3)
        melted_df['value']
        ax=axes[i][2]
        sns.lineplot(melted_df, x=X_COL, y='value', hue='variable', ax=ax)
        ax.grid()
        ax.set_ylabel('Count')
        if i < N_subsets - 1:
            ax.get_legend().remove()
        
        ax = axes[i][3]
        melted_df = traj_df.reset_index().melt(id_vars=[X_COL], value_vars=cols_4)
        sns.lineplot(melted_df, x=X_COL, y='value', hue='variable', ax=ax)
        ax.grid()
        ax.set_yscale('linear')
        ax.set_ylabel('Juice per Wei')
        if i < N_subsets - 1:
         ax.get_legend().remove()


        ax = axes[i][4]
        melted_df = traj_df.reset_index().melt(id_vars=[X_COL], value_vars=cols_5)
        sns.lineplot(melted_df, x=X_COL, y='value', hue='variable', ax=ax)
        ax.grid()
        ax.set_yscale('log')
        ax.set_ylabel('Gwei')
        if i < N_subsets - 1:
            ax.get_legend().remove()

        ax = axes[i][5]
        melted_df = traj_df.reset_index().melt(id_vars=[X_COL], value_vars=cols_6)
        sns.lineplot(melted_df, x=X_COL, y='value', hue='variable', ax=ax)
        ax.grid()
        # ax.set_yscale('log')
        ax.set_ylabel('Juice per Mana')
        if i < N_subsets - 1:
         ax.get_legend().remove()

        ax = axes[i][6]
        melted_df = traj_df.reset_index().melt(id_vars=[X_COL], value_vars=cols_7)
        sns.lineplot(melted_df, x=X_COL, y='value', hue='variable', ax=ax)
        ax.grid()
        # ax.set_yscale('log')
        ax.set_ylabel('Wei per Mana')
        if i < N_subsets - 1:
         ax.get_legend().remove()


    plt.show()