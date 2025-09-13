# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io

# 日本語フォントの文字化け対策
import japanize_matplotlib

# --- Streamlitアプリの基本設定 ---
st.set_page_config(page_title="全自動EDAレポートツール", page_icon="📝", layout="wide")
st.title("📝 全自動EDA（探索的データ分析）レポートツール")
st.write("ファイルをアップロードするだけで、データ分析レポートを自動生成します。")

# --- Session Stateの初期化 ---
if 'df' not in st.session_state:
    st.session_state.df = None

# --- ヘルパー関数 ---
def create_download_button(fig, file_name, label="このグラフをダウンロード"):
    """Matplotlibのグラフオブジェクトからダウンロードボタンを生成する"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=file_name,
        mime="image/png",
    )

# --- サイドバー ---
with st.sidebar:
    st.header("1. ファイルをアップロード")
    uploaded_file = st.file_uploader("CSVまたはExcelファイルをアップロード", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, parse_dates=True)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("ファイルが正常に読み込まれました！")
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")

# --- メイン画面 ---
if st.session_state.df is not None:
    df = st.session_state.df

    # ▼▼▼ セクション1: データ全体の概要と相関分析 ▼▼▼
    st.header("セクション1: データ全体の概要と相関分析")
    with st.expander("データプレビュー、基本情報などを表示", expanded=True):
        st.subheader("データプレビュー（先頭5行）")
        st.dataframe(df.head())
        st.subheader("基本情報")
        st.markdown(f"**行数:** {df.shape[0]} 行, **列数:** {df.shape[1]} 列")

    st.subheader("全体の相関分析")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        st.write("▼ 相関係数")
        corr_matrix = df[numeric_cols].corr()
        st.dataframe(corr_matrix)
        
        st.write("▼ ヒートマップ")
        fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax_corr)
        st.pyplot(fig_corr)
        create_download_button(fig_corr, "correlation_heatmap.png", "ヒートマップをダウンロード")
    else:
        st.info("相関分析を行うには、少なくとも2つ以上の数値列が必要です。")
    st.markdown("---")
    # ▲▲▲ セクション1ここまで ▲▲▲


    # ▼▼▼ セクション2: 全カラムの個別詳細分析 ▼▼▼
    st.header("セクション2: 全カラムの個別詳細分析")
    st.write("データフレームの全ての列について、データ型に応じた分析を自動で行います。")

    for col_name in df.columns:
        st.subheader(f"【 {col_name} 】列の分析結果", divider='blue')
        
        col1, col2 = st.columns([1, 2])

        # --- 数値データの場合 ---
        if pd.api.types.is_numeric_dtype(df[col_name]):
            with col1:
                st.write("**統計量**")
                stats_df = df[col_name].describe()
                stats_df['variance'] = df[col_name].var()
                st.dataframe(stats_df)

            with col2:
                # ▼▼▼ 変更点: 欠損値のみの列の場合、エラーを回避 ▼▼▼
                if df[col_name].dropna().empty:
                    st.write("**分布**")
                    st.info("この列は欠損値のみのため、グラフを描画できません。")
                else:
                    st.write("**分布（箱ひげ図とヒストグラム）**")
                    # ▼▼▼ 変更点: 箱ひげ図とヒストグラムを結合して表示 ▼▼▼
                    fig_dist, (ax_box, ax_hist) = plt.subplots(
                        2, 1, sharex=True, figsize=(8, 6),
                        gridspec_kw={"height_ratios": (.15, .85)}
                    )
                    # 上段に箱ひげ図
                    sns.boxplot(x=df[col_name], ax=ax_box)
                    ax_box.set_title(f'「{col_name}」の箱ひげ図とヒストグラム')
                    ax_box.set(xlabel='') # 上のグラフのx軸ラベルを消す
                    
                    # 下段にヒストグラム
                    sns.histplot(df[col_name], kde=True, ax=ax_hist)
                    ax_hist.set(xlabel='値') # x軸ラベルを共通で設定
                    
                    plt.subplots_adjust(hspace=0) # グラフ間の余白をなくす
                    st.pyplot(fig_dist)
                    create_download_button(fig_dist, f"distribution_{col_name}.png")
                    # ▲▲▲ 変更ここまで ▲▲▲

        # --- カテゴリデータ（文字列など）の場合 ---
        else:
            with col1:
                st.write("**統計量**")
                stats_df = df[col_name].describe()
                st.dataframe(stats_df)
                
                st.write("**カテゴリ別件数（上位10件）**")
                st.dataframe(df[col_name].value_counts().nlargest(10))

            with col2:
                st.write("**件数グラフ（上位20件）**")
                unique_count = df[col_name].nunique()
                if unique_count > 20:
                    st.warning(f"カテゴリ数が{unique_count}と多いため、グラフ描画を上位20件に制限します。")
                
                fig_count, ax_count = plt.subplots(figsize=(8, 6))
                sns.countplot(y=df[col_name], order=df[col_name].value_counts().nlargest(20).index, ax=ax_count)
                ax_count.set_title(f'カテゴリごとの件数（上位20件）')
                plt.tight_layout()
                st.pyplot(fig_count)
                create_download_button(fig_count, f"countplot_{col_name}.png")
        
        st.markdown("---")
    # ▲▲▲ セクション2ここまで ▲▲▲


    # ▼▼▼ セクション3: 時系列データの自動グラフ化 ▼▼▼
    st.header("セクション3: 時系列グラフ（該当列が存在する場合のみ）")
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()

    if not datetime_cols:
        st.info("データ内に日付・時刻形式の列が見つかりませんでした。")
    else:
        time_col = datetime_cols[0]
        st.success(f"時系列データ列 **`{time_col}`** を検知しました。これをX軸として、全ての数値列の折れ線グラフを自動生成します。")
        
        for num_col in numeric_cols:
            if num_col != time_col:
                st.subheader(f"時系列プロット: `{num_col}`")
                fig_line, ax_line = plt.subplots(figsize=(12, 5))
                sns.lineplot(x=df[time_col], y=df[num_col], ax=ax_line)
                ax_line.set_title(f'{time_col}に対する{num_col}の推移')
                ax_line.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig_line)
                create_download_button(fig_line, f"timeseries_{time_col}_vs_{num_col}.png")
    # ▲▲▲ セクション3ここまで ▲▲▲

else:
    st.info("サイドバーからファイル（CSVまたはExcel）をアップロードして分析を開始してください。")

